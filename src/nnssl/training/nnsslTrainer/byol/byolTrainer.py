"""
BYOL (Bootstrap Your Own Latent) Trainer for nnssl framework - FIXED & READY

CHANGES:
1. Removed broken 'LambdaTransform' import.
2. Added local 'SplitBYOLViews' class to handle view splitting.
3. Added 'BYOLTrainer_BS24_256iso' class to match your command.
"""

from copy import deepcopy
from typing import Union, Tuple, List
import math

import numpy as np
import torch
from torch import nn
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from torch import autocast
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.utilities.helpers import dummy_context

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper

# FIX: Removed LambdaTransform from imports
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

# IMPORT REPO TRANSFORMS
from nnssl.ssl_data.dataloading.simclr_transform import SimCLRTransform
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


# ============================================================================
# Helper Transform Class (Replaces LambdaTransform)
# ============================================================================

class SplitBYOLViews(AbstractTransform):
    """
    Splits the concatenated batch from SimCLRTransform into view1 and view2.
    """
    def __call__(self, **data_dict):
        # SimCLRTransform overwrites 'data' with [CropA_batch, CropB_batch]
        crops = data_dict['data'] 
        batch_size = data_dict['batch_size']
        
        # Split back into two views
        view1 = crops[:batch_size]
        view2 = crops[batch_size:]
        
        return {
            'view1': view1, 
            'view2': view2, 
            'batch_size': batch_size
        }


# ============================================================================
# BYOL Architecture Components
# ============================================================================

class BYOLProjectionHead(nn.Module):
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
    def __init__(
        self, 
        encoder: nn.Module, 
        features: list[int],
        hidden_dim: int = 4096,
        projection_dim: int = 256,
        tau_base: float = 0.996,
    ):
        super().__init__()
        
        total_features = sum(features) if isinstance(features, (list, tuple)) else features
        
        self.tau_base = tau_base
        self.tau = tau_base
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.online_encoder = encoder
        self.online_projector = BYOLProjectionHead(total_features, hidden_dim, projection_dim)
        self.online_predictor = BYOLPredictorHead(projection_dim, hidden_dim, projection_dim)
        
        self.target_encoder = None
        self.target_projector = None
        self._target_initialized = False
    
    def _init_target_network(self):
        if self._target_initialized:
            return
            
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projector = deepcopy(self.online_projector)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        self._target_initialized = True
    
    @torch.no_grad()
    def update_target_network(self, current_step: int = None, max_steps: int = None):
        if not self._target_initialized:
            self._init_target_network()
            return
        
        if current_step is not None and max_steps is not None and max_steps > 0:
            self.tau = 1 - (1 - self.tau_base) * (
                math.cos(math.pi * current_step / max_steps) + 1
            ) / 2
        
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
            pooled = [self.adaptive_pool(o) for o in out]
            flat = torch.cat(pooled, dim=1)
        else:
            flat = self.adaptive_pool(out)
        return flat.view(flat.shape[0], -1)
    
    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_and_pool(x, self.online_encoder)
        projected = self.online_projector(encoded)
        predicted = self.online_predictor(projected)
        return predicted
    
    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        if not self._target_initialized:
            self._init_target_network()
        encoded = self._encode_and_pool(x, self.target_encoder)
        projected = self.target_projector(encoded)
        return projected
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_and_pool(x, self.online_encoder)
        projected = self.online_projector(encoded)
        return projected


# ============================================================================
# BYOL Loss
# ============================================================================

class BYOLLoss(nn.Module):
    def forward(
        self,
        online_pred_1: torch.Tensor,
        online_pred_2: torch.Tensor,
        target_proj_1: torch.Tensor,
        target_proj_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        
        p1 = nn.functional.normalize(online_pred_1, dim=-1, p=2)
        p2 = nn.functional.normalize(online_pred_2, dim=-1, p=2)
        z1 = nn.functional.normalize(target_proj_1.detach(), dim=-1, p=2)
        z2 = nn.functional.normalize(target_proj_2.detach(), dim=-1, p=2)
        
        loss_1 = 2 - 2 * (p1 * z2).sum(dim=-1).mean()
        loss_2 = 2 - 2 * (p2 * z1).sum(dim=-1).mean()
        
        total_loss = (loss_1 + loss_2) / 2
        
        cos_sim = (p1 * z2).sum(dim=-1).mean().item()
        pseudo_acc = (cos_sim + 1) / 2
        
        return total_loss, pseudo_acc


# ============================================================================
# BYOL Trainer
# ============================================================================

class BYOLTrainer(AbstractBaseTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple = (256, 256, 256),
        crop_size: tuple = (96, 96, 96),
        hidden_dim: int = 4096,
        projection_dim: int = 256,
        tau_base: float = 0.996,
    ):
        plan.configurations[configuration_name].patch_size = patch_size
        self.patch_size = patch_size
        self.crop_size = crop_size

        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.tau_base = tau_base
        
        self.current_step = 0
        self.max_steps = self.num_epochs * self.num_iterations_per_epoch

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
            
            tr_transforms = []
            
            if do_dummy_2d_data_aug:
                raise NotImplementedError("Data should be isotropic for BYOL!")
            
            # 1. Use the Repo's robust SimCLRTransform
            tr_transforms.append(
                SimCLRTransform(
                    crop_size=self.crop_size,
                    aug="train",
                    crop_count_per_image=1, 
                    min_overlap_ratio=0.5,
                    data_key="data",
                )
            )
            
            # 2. Split 'data' into 'view1' and 'view2' using local class
            tr_transforms.append(SplitBYOLViews())
            
            # 3. Convert to Tensor
            tr_transforms.append(NumpyToTensor(["view1", "view2"], "float"))
            
            return Compose(tr_transforms)

    def get_validation_transforms(self) -> AbstractTransform:
        val_transforms = []
        
        # Validation Transform
        val_transforms.append(
            SimCLRTransform(
                crop_size=self.crop_size, 
                aug="val", 
                crop_count_per_image=1,
                data_key="data",
            )
        )
        
        # Split
        val_transforms.append(SplitBYOLViews())
        
        # To Tensor
        val_transforms.append(NumpyToTensor(["view1", "view2"], "float"))
        return Compose(val_transforms)

    def get_dataloaders(self):
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
        if hasattr(self.network, 'module'):
            return self.network.module
        return self.network

    def train_step(self, batch: dict) -> dict:
        view1 = batch["view1"].to(self.device, non_blocking=True)
        view2 = batch["view2"].to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        model = self._get_model()
        
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            online_pred_1 = model.forward_online(view1)
            online_pred_2 = model.forward_online(view2)
            
            with torch.no_grad():
                target_proj_1 = model.forward_target(view1)
                target_proj_2 = model.forward_target(view2)
            
            loss, acc = self.loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2)
        
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
        
        model.update_target_network(self.current_step, self.max_steps)
        self.current_step += 1
        
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
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
# Trainer Variants - DEFINED WITH CROP SIZES
# ============================================================================

class BYOLTrainer_BS8_256iso(BYOLTrainer):
    """BYOL for 256³ data, cropping to 96³."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256), 
            crop_size=(96, 96, 96),     
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 8 


class BYOLTrainer_BS16_256iso(BYOLTrainer):
    """BYOL with larger batch size (16), possible because crop is small (96³)."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            crop_size=(96, 96, 96),
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 16

class BYOLTrainer_BS24_256iso(BYOLTrainer):
    """BYOL with very large batch size (24). Warning: High VRAM usage."""
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(
            plan, configuration_name, fold, pretrain_json, device,
            patch_size=(256, 256, 256),
            crop_size=(96, 96, 96),
            hidden_dim=4096,
            projection_dim=256,
            tau_base=0.996,
        )
        self.total_batch_size = 24