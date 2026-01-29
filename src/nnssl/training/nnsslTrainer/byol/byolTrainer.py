"""
BYOL (Bootstrap Your Own Latent) Trainer for nnssl framework - CORRECTED VERSION

KEY INSIGHT: BYOL requires TWO AUGMENTED VIEWS OF THE SAME CONTENT.
This is different from SimCLR which uses overlapping crops at different positions.

The original BYOL paper:
"From an augmented view of an image, we train the online network to predict 
the target network representation of the same image under a DIFFERENT AUGMENTED VIEW"

So we need:
- Same patch/image
- Two DIFFERENT random augmentations applied
- NOT two crops from different spatial positions

This implementation:
1. Uses a BYOLTransform that creates two augmented views of the FULL patch
2. Follows true BYOL architecture (online+target, EMA updates)
3. Uses BYOL loss (prediction, no negatives)

Reference: "Bootstrap Your Own Latent" (Grill et al., 2020)
"""

from copy import deepcopy
from typing import Union, Tuple, List
import math

import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from einops import rearrange

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from torch import autocast
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

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


# ============================================================================
# BYOL Transform - Creates TWO AUGMENTED VIEWS of the SAME patch
# ============================================================================

class BYOLTransform(AbstractTransform):
    """
    BYOL-specific transform: Creates two differently-augmented views of the SAME patch.
    
    This is DIFFERENT from SimCLRTransform which creates overlapping crops at 
    different spatial positions. BYOL requires augmentation diversity, not spatial diversity.
    
    Output:
        view1: (B, C, D, H, W) - first augmented view
        view2: (B, C, D, H, W) - second augmented view (same content, different augmentation)
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        data_key: str = "data",
        # Augmentation probabilities
        p_flip: float = 0.5,
        p_rot90: float = 0.5,
        p_noise: float = 0.15,
        p_brightness: float = 0.15,
        p_contrast: float = 0.15,
        p_gamma: float = 0.15,
        # Augmentation ranges
        noise_variance: Tuple[float, float] = (0.0, 0.1),
        brightness_range: Tuple[float, float] = (0.75, 1.25),
        contrast_range: Tuple[float, float] = (0.75, 1.25),
        gamma_range: Tuple[float, float] = (0.7, 1.5),
    ):
        self.patch_size = patch_size
        self.data_key = data_key
        
        # Augmentation config
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
        self.p_noise = p_noise
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_gamma = p_gamma
        
        self.noise_variance = noise_variance
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
    
    def _augment_single(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a single volume.
        Each call produces a DIFFERENT random augmentation.
        
        Args:
            data: (C, D, H, W) volume
        Returns:
            augmented: (C, D, H, W) augmented volume
        """
        # Make a copy to avoid modifying original
        data = data.copy()
        
        # Random flips (independent for each axis)
        if np.random.random() < self.p_flip:
            axis = np.random.choice([1, 2, 3])  # D, H, or W axis
            data = np.flip(data, axis=axis).copy()
        
        # Random 90-degree rotations
        if np.random.random() < self.p_rot90:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            axes = [(1, 2), (1, 3), (2, 3)]  # rotation planes
            ax = axes[np.random.randint(0, 3)]
            data = np.rot90(data, k=k, axes=ax).copy()
        
        # Gaussian noise
        if np.random.random() < self.p_noise:
            variance = np.random.uniform(*self.noise_variance)
            noise = np.random.normal(0, np.sqrt(variance), data.shape).astype(np.float32)
            data = data + noise
        
        # Brightness (multiplicative)
        if np.random.random() < self.p_brightness:
            factor = np.random.uniform(*self.brightness_range)
            data = data * factor
        
        # Contrast
        if np.random.random() < self.p_contrast:
            factor = np.random.uniform(*self.contrast_range)
            mean = data.mean()
            data = (data - mean) * factor + mean
        
        # Gamma correction
        if np.random.random() < self.p_gamma:
            gamma = np.random.uniform(*self.gamma_range)
            data_min = data.min()
            data_range = data.max() - data_min
            if data_range > 1e-8:
                data = np.power((data - data_min) / data_range + 1e-8, gamma) * data_range + data_min
        
        return data.astype(np.float32)
    
    def __call__(self, **data_dict) -> dict:
        """
        Create two augmented views of each sample in the batch.
        
        Input: {"data": (B, C, D, H, W)}
        Output: {"view1": (B, C, D, H, W), "view2": (B, C, D, H, W), "batch_size": B}
        """
        data = data_dict[self.data_key]  # (B, C, D, H, W)
        batch_size = data.shape[0]
        
        view1_list = []
        view2_list = []
        
        for b in range(batch_size):
            sample = data[b]  # (C, D, H, W)
            
            # Create TWO DIFFERENT augmented views of the SAME patch
            # This is the key difference from SimCLR!
            v1 = self._augment_single(sample)
            v2 = self._augment_single(sample)  # Different random augmentation!
            
            view1_list.append(v1)
            view2_list.append(v2)
        
        view1 = np.stack(view1_list, axis=0)  # (B, C, D, H, W)
        view2 = np.stack(view2_list, axis=0)  # (B, C, D, H, W)
        
        return {
            "view1": view1,
            "view2": view2,
            "batch_size": batch_size,
        }


# ============================================================================
# BYOL Architecture Components
# ============================================================================

class BYOLProjectionHead(nn.Module):
    """
    BYOL projection head: Linear → BN → ReLU → Linear
    Per paper: 4096 hidden dim, 256 output dim
    """
    def __init__(self, input_dim: int, hidden_dim: int = 4096, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLPredictorHead(nn.Module):
    """
    BYOL predictor head (ONLY on online network, NOT on target).
    Same architecture as projector.
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLArchitecture(nn.Module):
    """
    BYOL Architecture.
    
    Online network:  encoder → projector → predictor → prediction
    Target network:  encoder → projector → projection (NO predictor!)
    
    Target network is EMA of online network.
    """
    
    def __init__(
        self, 
        encoder: nn.Module, 
        features: list[int],
        hidden_dim: int = 4096,
        projection_dim: int = 256,
        tau_base: float = 0.996,
    ):
        super().__init__()
        
        # Total features from all encoder stages
        total_features = sum(features) if isinstance(features, (list, tuple)) else features
        
        self.tau_base = tau_base
        self.tau = tau_base
        
        # Pooling layer (shared, no parameters)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Online network components
        self.online_encoder = encoder
        self.online_projector = BYOLProjectionHead(total_features, hidden_dim, projection_dim)
        self.online_predictor = BYOLPredictorHead(projection_dim, hidden_dim, projection_dim)
        
        # Target network (initialized as copy, updated via EMA)
        self.target_encoder = None
        self.target_projector = None
        self._target_initialized = False
    
    def _init_target_network(self):
        """Initialize target network as deep copy of online network."""
        if self._target_initialized:
            return
            
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projector = deepcopy(self.online_projector)
        
        # Target network does NOT receive gradients
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        self._target_initialized = True
    
    @torch.no_grad()
    def update_target_network(self, current_step: int = None, max_steps: int = None):
        """
        Update target network via EMA.
        
        τ increases from τ_base to 1.0 over training (cosine schedule).
        Higher τ = slower target updates = more stable.
        """
        if not self._target_initialized:
            self._init_target_network()
            return
        
        # Update tau with cosine schedule
        if current_step is not None and max_steps is not None and max_steps > 0:
            self.tau = 1 - (1 - self.tau_base) * (
                math.cos(math.pi * current_step / max_steps) + 1
            ) / 2
        
        # EMA update: target = τ*target + (1-τ)*online
        for online_p, target_p in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            target_p.data.mul_(self.tau).add_(online_p.data, alpha=1 - self.tau)
        
        for online_p, target_p in zip(
            self.online_projector.parameters(), 
            self.target_projector.parameters()
        ):
            target_p.data.mul_(self.tau).add_(online_p.data, alpha=1 - self.tau)
    
    def _encode_and_pool(self, x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """Encode input and pool to vector."""
        out = encoder(x)
        # Handle multi-scale outputs (list of feature maps per stage)
        if isinstance(out, (list, tuple)):
            pooled = [self.adaptive_pool(o) for o in out]
            flat = torch.cat(pooled, dim=1)
        else:
            flat = self.adaptive_pool(out)
        return flat.view(flat.shape[0], -1)
    
    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        """
        Online network forward pass.
        encoder → projector → predictor
        Returns: prediction (after predictor)
        """
        encoded = self._encode_and_pool(x, self.online_encoder)
        projected = self.online_projector(encoded)
        predicted = self.online_predictor(projected)
        return predicted
    
    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Target network forward pass (no gradients).
        encoder → projector (NO predictor!)
        Returns: projection (before predictor)
        """
        if not self._target_initialized:
            self._init_target_network()
        encoded = self._encode_and_pool(x, self.target_encoder)
        projected = self.target_projector(encoded)
        return projected
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward (online projection without predictor, for compatibility)."""
        encoded = self._encode_and_pool(x, self.online_encoder)
        projected = self.online_projector(encoded)
        return projected


# ============================================================================
# BYOL Loss
# ============================================================================

class BYOLLoss(nn.Module):
    """
    BYOL loss: Negative cosine similarity between predictions and targets.
    
    L = 2 - 2 * <normalize(prediction), normalize(target)>
    
    Symmetrized: both directions.
    """
    
    def forward(
        self,
        online_pred_1: torch.Tensor,
        online_pred_2: torch.Tensor,
        target_proj_1: torch.Tensor,
        target_proj_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute BYOL loss.
        
        BYOL symmetry:
        - online(view1) predicts target(view2)
        - online(view2) predicts target(view1)
        """
        # L2 normalize
        p1 = nn.functional.normalize(online_pred_1, dim=-1, p=2)
        p2 = nn.functional.normalize(online_pred_2, dim=-1, p=2)
        z1 = nn.functional.normalize(target_proj_1.detach(), dim=-1, p=2)
        z2 = nn.functional.normalize(target_proj_2.detach(), dim=-1, p=2)
        
        # Symmetric loss
        # Direction 1: online(view1) → target(view2)
        loss_1 = 2 - 2 * (p1 * z2).sum(dim=-1).mean()
        # Direction 2: online(view2) → target(view1)
        loss_2 = 2 - 2 * (p2 * z1).sum(dim=-1).mean()
        
        total_loss = (loss_1 + loss_2) / 2
        
        # Pseudo-accuracy for monitoring
        cos_sim = (p1 * z2).sum(dim=-1).mean().item()
        pseudo_acc = (cos_sim + 1) / 2  # Map [-1,1] to [0,1]
        
        return total_loss, pseudo_acc


# ============================================================================
# BYOL Trainer
# ============================================================================

class BYOLTrainer(AbstractBaseTrainer):
    """
    BYOL Trainer with CORRECT dual-view augmentation.
    
    Key difference from SimCLR:
    - Uses BYOLTransform: two augmented views of SAME patch
    - NOT SimCLRTransform: overlapping crops at different positions
    """

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple = (256, 256, 256),
        hidden_dim: int = 4096,
        projection_dim: int = 256,
        tau_base: float = 0.996,
    ):
        plan.configurations[configuration_name].patch_size = patch_size
        self.patch_size = patch_size

        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        
        # BYOL hyperparameters
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.tau_base = tau_base
        
        # Step tracking for EMA schedule
        self.current_step = 0
        self.max_steps = self.num_epochs * self.num_iterations_per_epoch

    def build_loss(self) -> nn.Module:
        """Build BYOL loss."""
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
            """
            Training transforms using Repo's SimCLRTransform for proper spatial augmentation.
            """
            tr_transforms = []
            
            if do_dummy_2d_data_aug:
                raise NotImplementedError("Data should be isotropic for BYOL!")
            
            # 1. Use the Repo's robust SimCLRTransform
            # This generates 2 views: 1 reference + 1 overlapping crop
            # Result is stored in 'all_crops' with shape (2*B, C, D, H, W)
            tr_transforms.append(
                SimCLRTransform(
                    crop_size=self.patch_size, 
                    aug="train",
                    crop_count_per_image=1, # 1 ref + 1 overlap = 2 views total per image
                    min_overlap_ratio=0.5,  # Ensure views share anatomy
                    data_key="data",
                )
            )
            
            # 2. Adapter: Split 'all_crops' into 'view1' and 'view2' for BYOL logic
            def split_crops(**data):
                crops = data['all_crops'] # Shape: (2*B, C, D, H, W)
                batch_size = data['batch_size']
                
                # SimCLRTransform concatenates [reference_crops, overlapping_crops]
                # So the first half is View 1, the second half is View 2
                view1 = crops[:batch_size]
                view2 = crops[batch_size:]
                
                return {
                    'view1': view1, 
                    'view2': view2, 
                    'batch_size': batch_size
                }

            tr_transforms.append(LambdaTransform(split_crops))
            
            # 3. Convert to Tensor
            tr_transforms.append(NumpyToTensor(["view1", "view2"], "float"))
            
            return Compose(tr_transforms)

    def get_validation_transforms(self) -> AbstractTransform:
        """Validation transforms (minimal augmentation)."""
        val_transforms = []
        
        val_transforms.append(
            BYOLTransform(
                patch_size=self.patch_size,
                data_key="data",
                # No augmentation for validation
                p_flip=0.0,
                p_rot90=0.0,
                p_noise=0.0,
                p_brightness=0.0,
                p_contrast=0.0,
                p_gamma=0.0,
            )
        )
        
        val_transforms.append(NumpyToTensor(["view1", "view2"], "float"))
        return Compose(val_transforms)

    def get_dataloaders(self):
        """Build dataloaders."""
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Warning: Dummy 2D aug not recommended for BYOL")
        
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, mirror_axes, do_dummy_2d_data_aug,
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
        """Build BYOL architecture."""
        encoder = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            encoder_only=True,
        )
        
        architecture = BYOLArchitecture(
            encoder=encoder,
            features=encoder.output_channels,
            hidden_dim=self.hidden_dim,
            projection_dim=self.projection_dim,
            tau_base=self.tau_base,
        )
        
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

    def _get_model(self):
        """Get underlying model (handles DDP wrapper)."""
        if hasattr(self.network, 'module'):
            return self.network.module
        return self.network

    def train_step(self, batch: dict) -> dict:
        """
        BYOL training step.
        
        Batch contains:
        - view1: (B, C, D, H, W) - first augmented view of each patch
        - view2: (B, C, D, H, W) - second augmented view of SAME patch
        
        BYOL forward:
        - online(view1) predicts target(view2)
        - online(view2) predicts target(view1)
        """
        view1 = batch["view1"].to(self.device, non_blocking=True)
        view2 = batch["view2"].to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        model = self._get_model()
        
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Online network forward (encoder → projector → predictor)
            online_pred_1 = model.forward_online(view1)
            online_pred_2 = model.forward_online(view2)
            
            # Target network forward (encoder → projector, NO predictor)
            # No gradients through target!
            with torch.no_grad():
                target_proj_1 = model.forward_target(view1)
                target_proj_2 = model.forward_target(view2)
            
            # BYOL loss (symmetric)
            loss, acc = self.loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2)
        
        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        # EMA update of target network (BYOL-specific!)
        model.update_target_network(self.current_step, self.max_steps)
        self.current_step += 1
        
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """Validation step."""
        view1 = batch["view1"].to(self.device, non_blocking=True)
        view2 = batch["view2"].to(self.device, non_blocking=True)
        
        model = self._get_model()
        
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                online_pred_1 = model.forward_online(view1)
                online_pred_2 = model.forward_online(view2)
                target_proj_1 = model.forward_target(view1)
                target_proj_2 = model.forward_target(view2)
                
                loss, acc = self.loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2)
        
        return {"loss": loss.detach().cpu().numpy()}


# ============================================================================
# Trainer Variants
# ============================================================================

class BYOLTrainer_BS8(BYOLTrainer):
    """BYOL with batch size 8."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class BYOLTrainer_BS4(BYOLTrainer):
    """BYOL with batch size 4."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 4


class BYOLTrainer_BS8_256iso(BYOLTrainer):
    """BYOL for 256³ isotropic LSFM data."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 8


class BYOLTrainer_BS4_256iso(BYOLTrainer):
    """BYOL for 256³ isotropic data, smaller batch."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 4


class BYOLTrainer_BS8_128iso(BYOLTrainer):
    """BYOL for 128³ isotropic data."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(128, 128, 128),
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 8
