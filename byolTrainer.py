"""
BYOL (Bootstrap Your Own Latent) Trainer for nnssl framework.

Based on:
- "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" (Grill et al., 2020)
- Zoomlin approach: Two augmented views of full 3D patches, 3D UNet backbone

Key differences from SimCLR:
- No negative pairs needed
- Asymmetric architecture: predictor only on online network
- Target network updated via EMA (not gradient descent)

To use: Add this file to /src/nnssl/training/nnsslTrainer/byol/
Then run: nnssl_train 1 noresample -tr BYOLTrainer_BS8_256iso -p nnsslPlans -num_gpus 2
"""

from copy import deepcopy
from typing import Union, Tuple, List
import math

import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.utilities.helpers import dummy_context

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform,
    MirrorTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


# ============================================================================
# MLP Projector and Predictor
# ============================================================================

class MLP(nn.Module):
    """
    MLP for projector and predictor in BYOL.
    Architecture: Linear -> BatchNorm -> ReLU -> Linear
    
    Note: No BatchNorm on final output layer (as per BYOL paper)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 4096, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# BYOL Architecture Wrapper
# ============================================================================

class BYOLArchitecture(nn.Module):
    """
    BYOL architecture wrapper for nnssl encoder.
    
    Online Network: Encoder -> Projector -> Predictor
    Target Network: Encoder -> Projector (EMA of online, no predictor)
    
    Args:
        encoder: The 3D UNet encoder (ResEncL from nnssl)
        encoder_output_channels: Output channels of encoder
        projection_dim: Output dimension of projector (default 256)
        hidden_dim: Hidden dimension of MLPs (default 4096)
        tau_base: Base EMA decay rate (default 0.996)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_channels: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        tau_base: float = 0.996,
    ):
        super().__init__()
        
        self.tau_base = tau_base
        self.tau = tau_base
        
        # Online network components
        self.online_encoder = encoder
        self.online_projector = MLP(encoder_output_channels, hidden_dim, projection_dim)
        self.online_predictor = MLP(projection_dim, hidden_dim, projection_dim)
        
        # Target network (deep copy, no gradients)
        self.target_encoder = deepcopy(encoder)
        self.target_projector = MLP(encoder_output_channels, hidden_dim, projection_dim)
        
        # Initialize target with online weights
        self._copy_online_to_target()
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        # Global average pooling for encoder output
        self.gap = nn.AdaptiveAvgPool3d(1)
    
    def _copy_online_to_target(self):
        """Initialize target network with online network weights."""
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
    
    @torch.no_grad()
    def update_target_network(self, current_step: int = None, max_steps: int = None):
        """
        Update target network with exponential moving average of online network.
        
        EMA update: ξ ← τξ + (1-τ)θ
        
        If step info provided, use cosine schedule for tau:
        τ = 1 - (1 - τ_base) * (cos(πk/K) + 1) / 2
        """
        if current_step is not None and max_steps is not None:
            # Cosine schedule: tau increases from tau_base towards 1
            self.tau = 1 - (1 - self.tau_base) * (math.cos(math.pi * current_step / max_steps) + 1) / 2
        else:
            self.tau = self.tau_base
        
        # Update encoder
        for online_p, target_p in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_p.data = self.tau * target_p.data + (1 - self.tau) * online_p.data
        
        # Update projector
        for online_p, target_p in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_p.data = self.tau * target_p.data + (1 - self.tau) * online_p.data
    
    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through online network (encoder -> projector -> predictor)."""
        features = self.online_encoder(x)  # (B, C, D, H, W)
        pooled = self.gap(features).view(features.size(0), -1)  # (B, C)
        projection = self.online_projector(pooled)
        prediction = self.online_predictor(projection)
        return prediction
    
    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through target network (encoder -> projector, no predictor)."""
        features = self.target_encoder(x)
        pooled = self.gap(features).view(features.size(0), -1)
        projection = self.target_projector(pooled)
        return projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward for compatibility - returns online encoder features."""
        return self.online_encoder(x)


# ============================================================================
# BYOL Transform for Full-Patch Augmentation
# ============================================================================

class BYOLTransform(AbstractTransform):
    """
    BYOL augmentation transform that creates two augmented views of the FULL patch.
    
    Unlike SimCLR which crops, BYOL (Zoomlin style) augments the entire 256³ patch
    twice with different random augmentations.
    
    Augmentations applied:
    - Random flips (mirror)
    - Random 90-degree rotations
    - Gaussian noise
    - Gaussian blur
    - Brightness/contrast adjustments
    - Gamma correction
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        p_flip: float = 0.5,
        p_rotation: float = 0.5,
        noise_variance: Tuple[float, float] = (0, 0.1),
        p_noise: float = 0.5,
        blur_sigma: Tuple[float, float] = (0.5, 1.0),
        p_blur: float = 0.5,
        brightness_range: Tuple[float, float] = (0.75, 1.25),
        p_brightness: float = 0.5,
        contrast_range: Tuple[float, float] = (0.75, 1.25),
        p_contrast: float = 0.5,
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        p_gamma: float = 0.5,
        data_key: str = "data",
    ):
        self.patch_size = patch_size
        self.data_key = data_key
        
        # Augmentation parameters
        self.p_flip = p_flip
        self.p_rotation = p_rotation
        self.noise_variance = noise_variance
        self.p_noise = p_noise
        self.blur_sigma = blur_sigma
        self.p_blur = p_blur
        self.brightness_range = brightness_range
        self.p_brightness = p_brightness
        self.contrast_range = contrast_range
        self.p_contrast = p_contrast
        self.gamma_range = gamma_range
        self.p_gamma = p_gamma
    
    def _augment_single(self, data: np.ndarray) -> np.ndarray:
        """Apply augmentations to a single volume."""
        # data shape: (C, D, H, W)
        
        # Random flips
        if np.random.random() < self.p_flip:
            axis = np.random.choice([1, 2, 3])  # D, H, or W
            data = np.flip(data, axis=axis).copy()
        
        # Random 90-degree rotations
        if np.random.random() < self.p_rotation:
            k = np.random.randint(1, 4)  # 1, 2, or 3 times 90 degrees
            axes = [(1, 2), (1, 3), (2, 3)]
            ax = axes[np.random.randint(0, 3)]
            data = np.rot90(data, k=k, axes=ax).copy()
        
        # Gaussian noise
        if np.random.random() < self.p_noise:
            variance = np.random.uniform(*self.noise_variance)
            noise = np.random.normal(0, np.sqrt(variance), data.shape)
            data = data + noise
        
        # Gaussian blur (approximated with scipy if available, else skip)
        if np.random.random() < self.p_blur:
            try:
                from scipy.ndimage import gaussian_filter
                sigma = np.random.uniform(*self.blur_sigma)
                for c in range(data.shape[0]):
                    data[c] = gaussian_filter(data[c], sigma=sigma)
            except ImportError:
                pass
        
        # Brightness
        if np.random.random() < self.p_brightness:
            factor = np.random.uniform(*self.brightness_range)
            data = data * factor
        
        # Contrast
        if np.random.random() < self.p_contrast:
            factor = np.random.uniform(*self.contrast_range)
            mean = data.mean()
            data = (data - mean) * factor + mean
        
        # Gamma
        if np.random.random() < self.p_gamma:
            gamma = np.random.uniform(*self.gamma_range)
            data_min = data.min()
            data_range = data.max() - data_min
            if data_range > 0:
                data = np.power((data - data_min) / data_range, gamma) * data_range + data_min
        
        return data.astype(np.float32)
    
    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        # data shape: (B, C, D, H, W)
        
        batch_size = data.shape[0]
        view1_list = []
        view2_list = []
        
        for b in range(batch_size):
            sample = data[b]  # (C, D, H, W)
            
            # Create two different augmented views of the SAME full patch
            view1 = self._augment_single(sample.copy())
            view2 = self._augment_single(sample.copy())
            
            view1_list.append(view1)
            view2_list.append(view2)
        
        # Stack views: view1 and view2 for each sample
        # Output shape: (2*B, C, D, H, W) where first B are view1, next B are view2
        all_views = np.concatenate([
            np.stack(view1_list, axis=0),
            np.stack(view2_list, axis=0)
        ], axis=0)
        
        data_dict["all_views"] = all_views
        data_dict["batch_size"] = batch_size
        
        return data_dict


# ============================================================================
# BYOL Loss Function
# ============================================================================

class BYOLLoss(nn.Module):
    """
    BYOL loss: Mean squared error between L2-normalized predictions and targets.
    
    L = ||q̄(z) - z̄'||² = 2 - 2 * <q̄(z), z̄'>
    
    where q̄ and z̄' are L2-normalized.
    """
    
    def forward(
        self,
        online_pred_1: torch.Tensor,
        online_pred_2: torch.Tensor,
        target_proj_1: torch.Tensor,
        target_proj_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetrized BYOL loss.
        
        Args:
            online_pred_1: Online predictions from view 1
            online_pred_2: Online predictions from view 2
            target_proj_1: Target projections from view 1
            target_proj_2: Target projections from view 2
            
        Returns:
            Scalar loss value
        """
        # L2 normalize
        online_pred_1 = nn.functional.normalize(online_pred_1, dim=-1, p=2)
        online_pred_2 = nn.functional.normalize(online_pred_2, dim=-1, p=2)
        target_proj_1 = nn.functional.normalize(target_proj_1, dim=-1, p=2)
        target_proj_2 = nn.functional.normalize(target_proj_2, dim=-1, p=2)
        
        # Symmetric loss: predict view2 from view1 and vice versa
        loss_1 = 2 - 2 * (online_pred_1 * target_proj_2).sum(dim=-1).mean()
        loss_2 = 2 - 2 * (online_pred_2 * target_proj_1).sum(dim=-1).mean()
        
        return loss_1 + loss_2


# ============================================================================
# BYOL Trainer
# ============================================================================

class BYOLTrainer(AbstractBaseTrainer):
    """
    BYOL Trainer for nnssl framework.
    
    Implements Bootstrap Your Own Latent for self-supervised pretraining
    on 3D medical/microscopy images using the ResEncL (nnU-Net) encoder.
    
    Key features:
    - No negative pairs (unlike SimCLR)
    - Target network with EMA updates
    - Asymmetric architecture (predictor only on online network)
    - Full-patch augmentation (no cropping)
    """
    
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple = (256, 256, 256),
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        tau_base: float = 0.996,
    ):
        # Set patch size in plan
        plan.configurations[configuration_name].patch_size = patch_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.tau_base = tau_base
        
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        
        # Training hyperparameters
        self.initial_lr = 3e-4
        self.weight_decay = 1.5e-6
        self.grad_clip = 1.0
        
        # EMA tracking
        self.max_steps = self.num_epochs * self.num_iterations_per_epoch
        self.current_step = 0
    
    def build_loss(self) -> nn.Module:
        """Build BYOL loss function."""
        return BYOLLoss()
    
    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
    ) -> AbstractTransform:
        """Build training augmentation pipeline."""
        tr_transforms = []
        
        if do_dummy_2d_data_aug:
            raise NotImplementedError("BYOL requires 3D data - no 2D dummy aug supported")
        
        # BYOL Transform: creates two augmented views of full patch
        tr_transforms.append(
            BYOLTransform(
                patch_size=patch_size,
                p_flip=0.5,
                p_rotation=0.5,
                noise_variance=(0, 0.1),
                p_noise=0.5,
                blur_sigma=(0.5, 1.0),
                p_blur=0.5,
                brightness_range=(0.75, 1.25),
                p_brightness=0.5,
                contrast_range=(0.75, 1.25),
                p_contrast=0.5,
                gamma_range=(0.7, 1.5),
                p_gamma=0.5,
                data_key="data",
            )
        )
        
        tr_transforms.append(NumpyToTensor(["all_views"], "float"))
        return Compose(tr_transforms)
    
    def get_validation_transforms(self) -> AbstractTransform:
        """Build validation augmentation pipeline (minimal augmentation)."""
        val_transforms = []
        
        val_transforms.append(
            BYOLTransform(
                patch_size=self.config_plan.patch_size,
                p_flip=0.0,
                p_rotation=0.0,
                p_noise=0.0,
                p_blur=0.0,
                p_brightness=0.0,
                p_contrast=0.0,
                p_gamma=0.0,
                data_key="data",
            )
        )
        
        val_transforms.append(NumpyToTensor(["all_views"], "float"))
        return Compose(val_transforms)
    
    def get_dataloaders(self):
        """Build training and validation dataloaders."""
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Warning: Dummy 2D aug not supported for BYOL")
        
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
        )
        val_transforms = self.get_validation_transforms()
        
        dl_tr, dl_val = self.get_plain_dataloaders(patch_size)
        
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val
    
    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> Tuple[nn.Module, AdaptationPlan]:
        """Build BYOL architecture with ResEncL encoder."""
        
        # Get the encoder from nnssl (same as SimCLR/VoCo)
        encoder = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            encoder_only=True,
        )
        
        # Wrap in BYOL architecture
        architecture = BYOLArchitecture(
            encoder=encoder,
            encoder_output_channels=encoder.output_channels,
            projection_dim=self.projection_dim,
            hidden_dim=self.hidden_dim,
            tau_base=self.tau_base,
        )
        
        # Create adaptation plan for downstream fine-tuning
        plan = deepcopy(self.plan)
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="online_encoder.stages",
            key_to_stem="online_encoder.stem",
            keys_to_in_proj=("online_encoder.stem.convs.0.conv", "online_encoder.stem.convs.0.all_modules.0"),
        )
        
        return architecture, adapt_plan
    
    def train_step(self, batch: dict) -> dict:
        """Single training step for BYOL."""
        all_views = batch["all_views"]
        batch_size = batch["batch_size"]
        
        all_views = all_views.to(self.device, non_blocking=True)
        
        # Split into view1 and view2
        view1 = all_views[:batch_size]  # First half
        view2 = all_views[batch_size:]  # Second half
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Online network forward (with predictor)
            online_pred_1 = self.network.forward_online(view1)
            online_pred_2 = self.network.forward_online(view2)
            
            # Target network forward (no predictor, no gradients)
            target_proj_1 = self.network.forward_target(view1)
            target_proj_2 = self.network.forward_target(view2)
            
            # BYOL loss (symmetric)
            loss = self.loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2)
        
        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
        
        # Update target network with EMA
        self.network.update_target_network(self.current_step, self.max_steps)
        self.current_step += 1
        
        return {"loss": loss.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        """Validation step for BYOL."""
        all_views = batch["all_views"]
        batch_size = batch["batch_size"]
        
        all_views = all_views.to(self.device, non_blocking=True)
        
        view1 = all_views[:batch_size]
        view2 = all_views[batch_size:]
        
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                online_pred_1 = self.network.forward_online(view1)
                online_pred_2 = self.network.forward_online(view2)
                target_proj_1 = self.network.forward_target(view1)
                target_proj_2 = self.network.forward_target(view2)
                
                loss = self.loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2)
        
        return {"loss": loss.detach().cpu().numpy()}


# ============================================================================
# Trainer Variants
# ============================================================================

class BYOLTrainer_BS8(BYOLTrainer):
    """BYOL with batch size 8."""
    
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class BYOLTrainer_BS8_256iso(BYOLTrainer):
    """BYOL for 256³ isotropic data (your LSFM setup)."""
    
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(256, 256, 256),
            projection_dim=256,
            hidden_dim=4096,
            tau_base=0.996,
        )
        self.total_batch_size = 8


class BYOLTrainer_BS4_256iso(BYOLTrainer):
    """BYOL for 256³ isotropic data with smaller batch size (if OOM)."""
    
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(256, 256, 256),
            projection_dim=256,
            hidden_dim=4096,
            tau_base=0.996,
        )
        self.total_batch_size = 4


class BYOLTrainer_BS8_128iso(BYOLTrainer):
    """BYOL for 128³ isotropic data (lighter memory footprint)."""
    
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(128, 128, 128),
            projection_dim=256,
            hidden_dim=4096,
            tau_base=0.996,
        )
        self.total_batch_size = 8
