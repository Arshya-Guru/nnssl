"""
BYOL (Bootstrap Your Own Latent) Trainer for nnssl framework - FIXED VERSION

Based on:
- "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" (Grill et al., 2020)

Key features:
1. Random 3D crops (128³ from 256³ patches) with minimum 0.2 overlap
2. ASYMMETRIC augmentation between views (as per BYOL paper):
   - View 1 (t): Mandatory Gaussian blur (p=1.0), NO solarization
   - View 2 (t'): Optional Gaussian blur (p=0.1), optional solarization (p=0.2)
3. Eager target network initialization
4. Proper EMA updates

Fixes from original buggy version:
- No early return in update_target_network (was skipping first EMA update)
- Consistent all_views format (no mixing with view1/view2)
- Validation uses light augmentation (not zero)
- DDP-safe model access via _get_network()
- FIXED: np.random.choice bug in _apply_rotation

To use: Replace your existing byolTrainer.py with this file
Then run: nnssl_train 1 noresample -tr BYOLTrainer_BS16_256iso -p nnsslPlans -num_gpus 2
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
from scipy.ndimage import gaussian_filter

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

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


# ============================================================================
# MLP Components
# ============================================================================

class BYOLProjectionHead(nn.Module):
    """MLP projector: Linear -> BN -> ReLU -> Linear"""
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
    """MLP predictor: Linear -> BN -> ReLU -> Linear (only in online network)"""
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


# ============================================================================
# BYOL Architecture
# ============================================================================

class BYOLArchitecture(nn.Module):
    """
    BYOL architecture wrapper for nnssl encoder.
    
    Online Network: Encoder -> Projector -> Predictor
    Target Network: Encoder -> Projector (EMA of online, NO predictor)
    """
    
    def __init__(
        self, 
        encoder: nn.Module, 
        features,
        hidden_dim: int = 4096,
        projection_dim: int = 256,
        tau_base: float = 0.996,
    ):
        super().__init__()
        
        self.tau_base = tau_base
        self.tau = tau_base
        
        if isinstance(features, (list, tuple)):
            total_features = sum(features)
        else:
            total_features = features
        
        self.total_features = total_features
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Online Network
        self.online_encoder = encoder
        self.online_projector = BYOLProjectionHead(total_features, hidden_dim, projection_dim)
        self.online_predictor = BYOLPredictorHead(projection_dim, hidden_dim, projection_dim)
        
        # Target Network - EAGER initialization (not lazy!)
        self.target_encoder = deepcopy(encoder)
        self.target_projector = deepcopy(self.online_projector)
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_target_network(self, current_step: int = None, max_steps: int = None):
        """
        EMA update: ξ ← τξ + (1-τ)θ
        
        NO early return - this was the bug that caused collapse!
        """
        if current_step is not None and max_steps is not None and max_steps > 0:
            self.tau = 1 - (1 - self.tau_base) * (
                math.cos(math.pi * current_step / max_steps) + 1
            ) / 2
        else:
            self.tau = self.tau_base
        
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
        out = encoder(x)
        if isinstance(out, (list, tuple)):
            pooled_features = []
            for scale_out in out:
                pooled = self.adaptive_pool(scale_out)
                pooled = pooled.view(pooled.size(0), -1)
                pooled_features.append(pooled)
            return torch.cat(pooled_features, dim=1)
        else:
            pooled = self.adaptive_pool(out)
            return pooled.view(pooled.size(0), -1)
    
    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_and_pool(x, self.online_encoder)
        projected = self.online_projector(encoded)
        predicted = self.online_predictor(projected)
        return predicted
    
    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_and_pool(x, self.target_encoder)
        projected = self.target_projector(encoded)
        return projected
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)


# ============================================================================
# BYOL Loss
# ============================================================================

class BYOLLoss(nn.Module):
    """BYOL loss with diagnostics for collapse detection."""
    
    def forward(
        self,
        online_pred_1: torch.Tensor,
        online_pred_2: torch.Tensor,
        target_proj_1: torch.Tensor,
        target_proj_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        
        p1 = nn.functional.normalize(online_pred_1, dim=-1, p=2)
        p2 = nn.functional.normalize(online_pred_2, dim=-1, p=2)
        z1 = nn.functional.normalize(target_proj_1, dim=-1, p=2)
        z2 = nn.functional.normalize(target_proj_2, dim=-1, p=2)
        
        cos_sim_1 = (p1 * z2).sum(dim=-1)
        cos_sim_2 = (p2 * z1).sum(dim=-1)
        
        loss_1 = (2 - 2 * cos_sim_1).mean()
        loss_2 = (2 - 2 * cos_sim_2).mean()
        
        total_loss = (loss_1 + loss_2) / 2
        
        with torch.no_grad():
            diagnostics = {
                'cos_sim': ((cos_sim_1.mean() + cos_sim_2.mean()) / 2).item(),
                'pred_std': online_pred_1.std().item(),
                'target_std': target_proj_1.std().item(),
                'pred_feat_std': online_pred_1.std(dim=0).mean().item(),
            }
        
        return total_loss, diagnostics


# ============================================================================
# BYOL Transform with Random Crops and Asymmetric Augmentation
# ============================================================================

class BYOLTransform(AbstractTransform):
    """
    BYOL augmentation following the original paper's approach:
    
    1. Random 3D crops (crop_size from patch_size) with minimum overlap constraint
    2. ASYMMETRIC augmentation between views:
       - View 1 (t): Mandatory Gaussian blur (p=1.0), NO solarization
       - View 2 (t'): Optional Gaussian blur (p=0.1), optional solarization (p=0.2)
    
    This follows the SimCLR augmentation pipeline adapted for 3D:
    - Random crop (CRITICAL - this is the core augmentation)
    - Random flip
    - Random rotation
    - Intensity jitter (brightness, contrast, gamma) - analog to color jitter
    - Gaussian blur (asymmetric)
    - Solarization/intensity inversion (asymmetric, view 2 only)
    - Gaussian noise
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (256, 256, 256),
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        min_overlap_ratio: float = 0.2,
        # Shared augmentation probabilities
        p_flip: float = 0.5,
        p_rotation: float = 0.5,
        # Intensity jitter
        p_intensity_jitter: float = 0.8,
        brightness_range: Tuple[float, float] = (0.6, 1.4),
        contrast_range: Tuple[float, float] = (0.6, 1.4),
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        # Gaussian noise
        p_noise: float = 0.5,
        noise_variance: Tuple[float, float] = (0, 0.05),
        # Gaussian blur - ASYMMETRIC
        blur_sigma_range: Tuple[float, float] = (0.5, 2.0),
        p_blur_view1: float = 1.0,   # Mandatory for view 1
        p_blur_view2: float = 0.1,   # Optional for view 2
        # Solarization - ASYMMETRIC  
        p_solarization_view1: float = 0.0,  # Never for view 1
        p_solarization_view2: float = 0.2,  # Optional for view 2
        solarization_threshold: float = 0.5,
        # Keys
        data_key: str = "data",
    ):
        self.patch_size = np.array(patch_size)
        self.crop_size = np.array(crop_size)
        self.min_overlap_ratio = min_overlap_ratio
        
        self.p_flip = p_flip
        self.p_rotation = p_rotation
        
        self.p_intensity_jitter = p_intensity_jitter
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        
        self.p_noise = p_noise
        self.noise_variance = noise_variance
        
        self.blur_sigma_range = blur_sigma_range
        self.p_blur_view1 = p_blur_view1
        self.p_blur_view2 = p_blur_view2
        
        self.p_solarization_view1 = p_solarization_view1
        self.p_solarization_view2 = p_solarization_view2
        self.solarization_threshold = solarization_threshold
        
        self.data_key = data_key
        
        # Compute valid crop range
        self.max_start = self.patch_size - self.crop_size
        
        # Pre-define rotation axis options (for 3D: rotate in xy, xz, or yz plane)
        self._rotation_axes = [(1, 2), (1, 3), (2, 3)]
    
    def _get_random_crop_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two random crop coordinates with minimum overlap constraint."""
        # First crop: completely random
        start1 = np.array([
            np.random.randint(0, self.max_start[i] + 1) for i in range(3)
        ])
        
        # Second crop: constrained to have minimum overlap with first
        min_overlap_voxels = (self.min_overlap_ratio * self.crop_size).astype(int)
        
        start2 = np.zeros(3, dtype=int)
        for i in range(3):
            low = max(0, start1[i] - self.crop_size[i] + min_overlap_voxels[i])
            high = min(self.max_start[i], start1[i] + self.crop_size[i] - min_overlap_voxels[i])
            
            if low > high:
                low = max(0, start1[i] - self.crop_size[i] // 2)
                high = min(self.max_start[i], start1[i] + self.crop_size[i] // 2)
            
            if low > high:
                start2[i] = start1[i]
            else:
                start2[i] = np.random.randint(low, high + 1)
        
        return start1, start2
    
    def _extract_crop(self, volume: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Extract a crop from the volume."""
        return volume[
            :,
            start[0]:start[0] + self.crop_size[0],
            start[1]:start[1] + self.crop_size[1],
            start[2]:start[2] + self.crop_size[2],
        ].copy()
    
    def _apply_flip(self, volume: np.ndarray) -> np.ndarray:
        """Random flip along each spatial axis."""
        for axis in [1, 2, 3]:
            if np.random.random() < self.p_flip:
                volume = np.flip(volume, axis=axis)
        return volume.copy()
    
    def _apply_rotation(self, volume: np.ndarray) -> np.ndarray:
        """Random 90-degree rotation."""
        if np.random.random() < self.p_rotation:
            k = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            # FIX: np.random.choice can't select from list of tuples
            # Use randint to index into the list instead
            axes = self._rotation_axes[np.random.randint(len(self._rotation_axes))]
            volume = np.rot90(volume, k=k, axes=axes)
        return volume.copy()
    
    def _apply_intensity_jitter(self, volume: np.ndarray) -> np.ndarray:
        """Apply intensity jitter (brightness, contrast, gamma in random order)."""
        if np.random.random() >= self.p_intensity_jitter:
            return volume
        
        transforms = ['brightness', 'contrast', 'gamma']
        np.random.shuffle(transforms)
        
        for t in transforms:
            if t == 'brightness':
                factor = np.random.uniform(*self.brightness_range)
                volume = volume * factor
            elif t == 'contrast':
                factor = np.random.uniform(*self.contrast_range)
                mean_val = volume.mean()
                volume = (volume - mean_val) * factor + mean_val
            elif t == 'gamma':
                gamma = np.random.uniform(*self.gamma_range)
                min_val = volume.min()
                volume = np.power(volume - min_val + 1e-8, gamma) + min_val
        
        return volume
    
    def _apply_gaussian_blur(self, volume: np.ndarray, p_blur: float) -> np.ndarray:
        """Apply Gaussian blur with given probability."""
        if np.random.random() >= p_blur:
            return volume
        
        sigma = np.random.uniform(*self.blur_sigma_range)
        for c in range(volume.shape[0]):
            volume[c] = gaussian_filter(volume[c], sigma=sigma)
        
        return volume
    
    def _apply_solarization(self, volume: np.ndarray, p_solar: float) -> np.ndarray:
        """Apply solarization (intensity inversion above threshold)."""
        if np.random.random() >= p_solar:
            return volume
        
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min < 1e-8:
            return volume
        
        volume_norm = (volume - v_min) / (v_max - v_min)
        mask = volume_norm > self.solarization_threshold
        volume_norm[mask] = 2 * self.solarization_threshold - volume_norm[mask]
        volume = volume_norm * (v_max - v_min) + v_min
        
        return volume
    
    def _apply_noise(self, volume: np.ndarray) -> np.ndarray:
        """Apply additive Gaussian noise."""
        if np.random.random() >= self.p_noise:
            return volume
        
        var = np.random.uniform(*self.noise_variance)
        if var > 0:
            noise = np.random.normal(0, np.sqrt(var), volume.shape).astype(np.float32)
            volume = volume + noise
        
        return volume
    
    def _augment_view(
        self, 
        crop: np.ndarray, 
        p_blur: float, 
        p_solar: float
    ) -> np.ndarray:
        """Apply full augmentation pipeline to a crop."""
        result = crop.copy()
        
        # Spatial augmentations
        result = self._apply_flip(result)
        result = self._apply_rotation(result)
        
        # Intensity augmentations
        result = self._apply_intensity_jitter(result)
        result = self._apply_gaussian_blur(result, p_blur)
        result = self._apply_solarization(result, p_solar)
        result = self._apply_noise(result)
        
        return result.astype(np.float32)
    
    def __call__(self, **data_dict):
        data = data_dict[self.data_key]  # Shape: (B, C, D, H, W)
        batch_size = data.shape[0]
        
        view1_list = []
        view2_list = []
        
        for b in range(batch_size):
            sample = data[b]  # Shape: (C, D, H, W)
            
            # Get two random crop coordinates with overlap constraint
            start1, start2 = self._get_random_crop_coords()
            
            # Extract crops
            crop1 = self._extract_crop(sample, start1)
            crop2 = self._extract_crop(sample, start2)
            
            # Apply ASYMMETRIC augmentation
            # View 1: Mandatory blur (p=1.0), NO solarization (p=0.0)
            view1 = self._augment_view(
                crop1, 
                p_blur=self.p_blur_view1,
                p_solar=self.p_solarization_view1
            )
            
            # View 2: Optional blur (p=0.1), optional solarization (p=0.2)
            view2 = self._augment_view(
                crop2,
                p_blur=self.p_blur_view2,
                p_solar=self.p_solarization_view2
            )
            
            view1_list.append(view1)
            view2_list.append(view2)
        
        # Stack: [view1_batch, view2_batch] → (2*B, C, crop_D, crop_H, crop_W)
        all_views = np.concatenate([
            np.stack(view1_list, axis=0),
            np.stack(view2_list, axis=0)
        ], axis=0)
        
        data_dict["all_views"] = all_views
        data_dict["batch_size"] = batch_size
        
        return data_dict


# ============================================================================
# BYOL Trainer
# ============================================================================

class BYOLTrainer(AbstractBaseTrainer):
    """
    BYOL Trainer for nnssl framework.
    
    Uses random crops with asymmetric augmentation as per the original paper.
    """
    
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: Tuple[int, int, int] = (256, 256, 256),
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        min_overlap_ratio: float = 0.2,
        hidden_dim: int = 4096,
        projection_dim: int = 256,
        tau_base: float = 0.996,
    ):
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.min_overlap_ratio = min_overlap_ratio
        
        plan.configurations[configuration_name].patch_size = patch_size
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.tau_base = tau_base
        
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        
        self.initial_lr = 3e-4
        self.weight_decay = 1.5e-6
        self.grad_clip = 1.0
        
        self.current_step = 0
        self.max_steps = self.num_epochs * self.num_iterations_per_epoch
        
        self._collapse_warning_count = 0
    
    def build_loss(self) -> nn.Module:
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
        """Build training augmentation pipeline with random crops."""
        
        if do_dummy_2d_data_aug:
            raise NotImplementedError("BYOL requires 3D isotropic data!")
        
        tr_transforms = []
        
        tr_transforms.append(
            BYOLTransform(
                patch_size=self.patch_size,
                crop_size=self.crop_size,
                min_overlap_ratio=self.min_overlap_ratio,
                p_flip=0.5,
                p_rotation=0.5,
                p_intensity_jitter=0.8,
                brightness_range=(0.6, 1.4),
                contrast_range=(0.6, 1.4),
                gamma_range=(0.7, 1.5),
                p_noise=0.5,
                noise_variance=(0, 0.05),
                blur_sigma_range=(0.5, 2.0),
                p_blur_view1=1.0,
                p_blur_view2=0.1,
                p_solarization_view1=0.0,
                p_solarization_view2=0.2,
                data_key="data",
            )
        )
        
        tr_transforms.append(NumpyToTensor(["all_views"], "float"))
        
        return Compose(tr_transforms)
    
    def get_validation_transforms(self) -> AbstractTransform:
        """Validation with LIGHT augmentation (not zero!)."""
        val_transforms = []
        
        val_transforms.append(
            BYOLTransform(
                patch_size=self.patch_size,
                crop_size=self.crop_size,
                min_overlap_ratio=0.5,
                p_flip=0.3,
                p_rotation=0.3,
                p_intensity_jitter=0.5,
                brightness_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2),
                gamma_range=(0.9, 1.1),
                p_noise=0.2,
                noise_variance=(0, 0.02),
                blur_sigma_range=(0.5, 1.5),
                p_blur_view1=1.0,
                p_blur_view2=0.1,
                p_solarization_view1=0.0,
                p_solarization_view2=0.1,
                data_key="data",
            )
        )
        
        val_transforms.append(NumpyToTensor(["all_views"], "float"))
        
        return Compose(val_transforms)
    
    def get_dataloaders(self):
        """Build dataloaders - loads full patches, transform does cropping."""
        patch_size = self.patch_size
        
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
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
        """Build BYOL architecture - network sees crop_size, not patch_size."""
        
        # Network architecture uses CROP size
        crop_config_plan = deepcopy(config_plan)
        crop_config_plan.patch_size = self.crop_size
        
        encoder = get_network_by_name(
            crop_config_plan,
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
            recommended_downstream_patchsize=self.crop_size,
            pretrain_num_input_channels=1,
            key_to_encoder="online_encoder.stages",
            key_to_stem="online_encoder.stem",
            keys_to_in_proj=(
                "online_encoder.stem.convs.0.conv",
                "online_encoder.stem.convs.0.all_modules.0"
            ),
        )
        
        return architecture, adapt_plan
    
    def _get_network(self) -> nn.Module:
        """Get underlying network (handles DDP wrapper)."""
        if hasattr(self.network, 'module'):
            return self.network.module
        return self.network
    
    def train_step(self, batch: dict) -> dict:
        """Single training step for BYOL."""
        all_views = batch["all_views"]
        batch_size = batch["batch_size"]
        
        all_views = all_views.to(self.device, non_blocking=True)
        
        view1 = all_views[:batch_size]
        view2 = all_views[batch_size:]
        
        self.optimizer.zero_grad(set_to_none=True)
        
        network = self._get_network()
        
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            online_pred_1 = network.forward_online(view1)
            online_pred_2 = network.forward_online(view2)
            
            target_proj_1 = network.forward_target(view1)
            target_proj_2 = network.forward_target(view2)
            
            loss, diagnostics = self.loss(
                online_pred_1, online_pred_2,
                target_proj_1, target_proj_2
            )
        
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
        
        # EMA update AFTER optimizer step
        network.update_target_network(self.current_step, self.max_steps)
        self.current_step += 1
        
        # Collapse detection
        if diagnostics['cos_sim'] > 0.99 or diagnostics['pred_std'] < 0.01:
            self._collapse_warning_count += 1
            if self._collapse_warning_count <= 5:
                self.print_to_log_file(
                    f"WARNING: Possible collapse! cos_sim={diagnostics['cos_sim']:.4f}, "
                    f"pred_std={diagnostics['pred_std']:.6f}"
                )
        
        return {
            "loss": loss.detach().cpu().numpy(),
            "cos_sim": diagnostics['cos_sim'],
            "pred_std": diagnostics['pred_std'],
        }
    
    def validation_step(self, batch: dict) -> dict:
        """Validation step for BYOL."""
        all_views = batch["all_views"]
        batch_size = batch["batch_size"]
        
        all_views = all_views.to(self.device, non_blocking=True)
        
        view1 = all_views[:batch_size]
        view2 = all_views[batch_size:]
        
        network = self._get_network()
        
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                online_pred_1 = network.forward_online(view1)
                online_pred_2 = network.forward_online(view2)
                target_proj_1 = network.forward_target(view1)
                target_proj_2 = network.forward_target(view2)
                
                loss, diagnostics = self.loss(
                    online_pred_1, online_pred_2,
                    target_proj_1, target_proj_2
                )
        
        return {
            "loss": loss.detach().cpu().numpy(),
            "cos_sim": diagnostics['cos_sim'],
        }


# ============================================================================
# Trainer Variants
# ============================================================================

class BYOLTrainer_BS8(BYOLTrainer):
    """BYOL with batch size 8, 256³ patches → 128³ crops."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            crop_size=(128, 128, 128),
            min_overlap_ratio=0.2,
        )
        self.total_batch_size = 8


class BYOLTrainer_BS8_256iso(BYOLTrainer):
    """BYOL for 256³ patches → 128³ crops."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            crop_size=(128, 128, 128),
            min_overlap_ratio=0.2,
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 8


class BYOLTrainer_BS8_256iso_96crop(BYOLTrainer):
    """BYOL for 256³ patches → 128³ crops."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            crop_size=(96, 96, 96),
            min_overlap_ratio=0.2,
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 8


class BYOLTrainer_BS16_256iso(BYOLTrainer):
    """BYOL with batch size 16."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            crop_size=(128, 128, 128),
            min_overlap_ratio=0.2,
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 16


class BYOLTrainer_BS4_256iso(BYOLTrainer):
    """BYOL with smaller batch size (if OOM)."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            crop_size=(128, 128, 128),
            min_overlap_ratio=0.2,
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 4


class BYOLTrainer_BS8_128iso(BYOLTrainer):
    """BYOL for smaller patches: 128³ → 64³ crops."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(128, 128, 128),
            crop_size=(64, 64, 64),
            min_overlap_ratio=0.2,
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 8