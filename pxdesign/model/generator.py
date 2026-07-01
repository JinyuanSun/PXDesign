# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from protenix.model.utils import centre_random_augmentation


def apply_masked_guidance(
    x: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if weight <= 0:
        return x
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return x
    guided = x + float(weight) * (reference - x)
    return torch.where(mask.expand(*x.shape[:-1], 1), guided, x)


def initialize_masked_coords(
    x: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    sigma: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return x
    initialized = reference + sigma * noise
    return torch.where(mask.expand(*x.shape[:-1], 1), initialized, x)


def align_reference_to_masked_coords(
    reference: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return reference

    original_shape = coords.shape
    ref_flat = reference.reshape(-1, original_shape[-2], 3)
    coords_flat = coords.reshape(-1, original_shape[-2], 3)
    mask_flat = mask.reshape(-1, original_shape[-2], 1).to(dtype=coords.dtype)

    counts = mask_flat.sum(dim=-2, keepdim=True).clamp_min(1.0)
    ref_centroid = (ref_flat * mask_flat).sum(dim=-2, keepdim=True) / counts
    coords_centroid = (coords_flat * mask_flat).sum(dim=-2, keepdim=True) / counts
    ref_centered = ref_flat - ref_centroid
    coords_centered = coords_flat - coords_centroid

    covariance = torch.matmul(
        (ref_centered * mask_flat).transpose(-1, -2),
        coords_centered * mask_flat,
    )
    u, _, vh = torch.linalg.svd(covariance)
    v = vh.transpose(-1, -2)
    correction = torch.ones(
        (*v.shape[:-2], 3), device=coords.device, dtype=coords.dtype
    )
    determinant = torch.linalg.det(torch.matmul(u, vh))
    correction[..., -1] = torch.where(
        determinant < 0,
        torch.full_like(determinant, -1.0),
        torch.ones_like(determinant),
    )
    correction = torch.diag_embed(correction)
    rotation = torch.matmul(torch.matmul(u, correction), vh)
    aligned = torch.matmul(ref_centered, rotation) + coords_centroid
    return aligned.reshape(original_shape)


def apply_interchain_clash_guidance(
    x: torch.Tensor,
    target_mask: torch.Tensor,
    design_mask: torch.Tensor,
    threshold: float,
    weight: float,
) -> torch.Tensor:
    if weight <= 0 or threshold <= 0:
        return x
    target_mask = target_mask.bool()
    design_mask = design_mask.bool()
    if int(target_mask.sum().item()) == 0 or int(design_mask.sum().item()) == 0:
        return x

    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-2], 3)
    target_mask_flat = target_mask.reshape(-1, original_shape[-2])
    design_mask_flat = design_mask.reshape(-1, original_shape[-2])
    guided_flat = x_flat.clone()

    for sample_idx in range(x_flat.shape[0]):
        target_coords = x_flat[sample_idx, target_mask_flat[sample_idx]]
        design_indices = torch.nonzero(design_mask_flat[sample_idx], as_tuple=False).squeeze(-1)
        design_coords = x_flat[sample_idx, design_indices]
        if target_coords.numel() == 0 or design_coords.numel() == 0:
            continue

        diff = design_coords[:, None, :] - target_coords[None, :, :]
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
        close = dist < float(threshold)
        if not bool(close.any().item()):
            continue
        direction = diff / dist[..., None]
        penetration = (float(threshold) - dist).clamp_min(0.0)
        repel = direction * penetration[..., None] * close[..., None].to(dtype=x.dtype)
        n_close = close.sum().clamp_min(1).to(dtype=x.dtype)
        repel = repel.sum(dim=(0, 1)) / n_close
        guided_flat[sample_idx, design_indices] = (
            design_coords + float(weight) * repel
        )

    return guided_flat.reshape(original_shape)


def apply_hotspot_contact_guidance(
    x: torch.Tensor,
    cdr_mask: torch.Tensor,
    hotspot_mask: torch.Tensor,
    move_mask: torch.Tensor,
    target_distance: float,
    weight: float,
) -> torch.Tensor:
    if weight <= 0 or target_distance <= 0:
        return x
    cdr_mask = cdr_mask.bool()
    hotspot_mask = hotspot_mask.bool()
    move_mask = move_mask.bool()
    if (
        int(cdr_mask.sum().item()) == 0
        or int(hotspot_mask.sum().item()) == 0
        or int(move_mask.sum().item()) == 0
    ):
        return x

    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-2], 3)
    cdr_mask_flat = cdr_mask.reshape(-1, original_shape[-2])
    hotspot_mask_flat = hotspot_mask.reshape(-1, original_shape[-2])
    move_mask_flat = move_mask.reshape(-1, original_shape[-2])
    guided_flat = x_flat.clone()

    for sample_idx in range(x_flat.shape[0]):
        cdr_coords = x_flat[sample_idx, cdr_mask_flat[sample_idx]]
        hotspot_coords = x_flat[sample_idx, hotspot_mask_flat[sample_idx]]
        move_indices = torch.nonzero(move_mask_flat[sample_idx], as_tuple=False).squeeze(-1)
        if cdr_coords.numel() == 0 or hotspot_coords.numel() == 0 or move_indices.numel() == 0:
            continue

        diff = hotspot_coords[None, :, :] - cdr_coords[:, None, :]
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
        min_index = int(torch.argmin(dist).item())
        hotspot_count = hotspot_coords.shape[0]
        cdr_idx = min_index // hotspot_count
        hotspot_idx = min_index % hotspot_count
        min_dist = dist[cdr_idx, hotspot_idx]
        if min_dist <= float(target_distance):
            continue
        direction = diff[cdr_idx, hotspot_idx] / min_dist
        displacement = direction * (min_dist - float(target_distance))
        guided_flat[sample_idx, move_indices] = (
            x_flat[sample_idx, move_indices] + float(weight) * displacement
        )

    return guided_flat.reshape(original_shape)


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    # step_scale_eta: float = 1.5,
    step_scale_eta: Union[float, dict] = {"type": "const", "min": 1.5, "max": 1.5},
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    framework_init_noise_sigma: Optional[float] = None,
    framework_guidance_weight: float = 0.0,
    condition_init_from_coords: bool = False,
    clash_guidance_weight: float = 0.0,
    clash_guidance_threshold: float = 2.5,
    hotspot_guidance_weight: float = 0.0,
    hotspot_guidance_target_distance: float = 10.0,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    print("sampling eta schedule: ", step_scale_eta)

    label_dict = input_feature_dict.get("label_dict", {})
    condition_coordinate_mask = label_dict.get("condition_coordinate_mask")
    rigid_framework_coordinate = label_dict.get("rigid_framework_coordinate")
    rigid_framework_coordinate_mask = label_dict.get("rigid_framework_coordinate_mask")
    has_framework_init = (
        rigid_framework_coordinate is not None
        and rigid_framework_coordinate_mask is not None
        and int(rigid_framework_coordinate_mask.sum().item()) > 0
    )
    if (
        has_framework_init
        and framework_init_noise_sigma is not None
        and float(framework_init_noise_sigma) > 0
    ):
        diffs = (noise_schedule - float(framework_init_noise_sigma)).abs()
        start_idx = int(diffs.argmin().item())
        noise_schedule = noise_schedule[start_idx:]
        print(
            "framework init noise sigma: "
            f"{float(framework_init_noise_sigma):.3f}; "
            f"starting at schedule index {start_idx}, "
            f"sigma={float(noise_schedule[0]):.3f}, "
            f"steps={len(noise_schedule) - 1}"
        )

    def _framework_coords_and_mask(x):
        if rigid_framework_coordinate is None or rigid_framework_coordinate_mask is None:
            return None, None
        coords = rigid_framework_coordinate.to(device=device, dtype=dtype)
        mask = rigid_framework_coordinate_mask.to(device=device).bool()
        if int(mask.sum().item()) == 0:
            return None, None
        coords = coords.reshape(*([1] * len(batch_shape)), 1, N_atom, 3)
        mask = mask.reshape(*([1] * len(batch_shape)), 1, N_atom, 1)
        return coords.expand_as(x), mask.expand(*x.shape[:-1], 1)

    def _condition_coords_and_mask(x):
        if condition_coordinate_mask is None:
            return None, None
        condition_coordinate = label_dict.get("condition_coordinate")
        if condition_coordinate is None:
            return None, None
        coords = condition_coordinate.to(device=device, dtype=dtype)
        mask = condition_coordinate_mask.to(device=device).bool()
        if int(mask.sum().item()) == 0:
            return None, None
        coords = coords.reshape(*([1] * len(batch_shape)), 1, N_atom, 3)
        mask = mask.reshape(*([1] * len(batch_shape)), 1, N_atom, 1)
        return coords.expand_as(x), mask.expand(*x.shape[:-1], 1)

    def _initialize_condition_coords(x, sigma):
        if not condition_init_from_coords:
            return x
        coords, mask = _condition_coords_and_mask(x)
        if coords is None or mask is None:
            return x
        condition_noise = torch.randn(size=x.shape, device=device, dtype=dtype)
        return initialize_masked_coords(
            x=x,
            reference=coords,
            mask=mask,
            sigma=sigma,
            noise=condition_noise,
        )

    def _apply_framework_guidance(x, sigma, framework_noise):
        coords, mask = _framework_coords_and_mask(x)
        if coords is None or mask is None:
            return x
        reference = coords + sigma * framework_noise
        reference = align_reference_to_masked_coords(reference, x, mask)
        return apply_masked_guidance(
            x=x,
            reference=reference,
            mask=mask,
            weight=framework_guidance_weight,
        )

    def _target_and_design_masks(x):
        if condition_coordinate_mask is None:
            return None, None
        target_mask = condition_coordinate_mask.to(device=device).bool()
        target_mask = target_mask.reshape(*([1] * len(batch_shape)), 1, N_atom, 1)
        target_mask = target_mask.expand(*x.shape[:-1], 1)
        design_mask = ~target_mask
        return target_mask, design_mask

    def _framework_mask_for_shape(x):
        _, framework_mask = _framework_coords_and_mask(x)
        return framework_mask

    def _apply_clash_guidance(x):
        target_mask, design_mask = _target_and_design_masks(x)
        if target_mask is None or design_mask is None:
            return x
        return apply_interchain_clash_guidance(
            x=x,
            target_mask=target_mask,
            design_mask=design_mask,
            threshold=clash_guidance_threshold,
            weight=clash_guidance_weight,
        )

    def _apply_hotspot_guidance(x):
        target_mask, design_mask = _target_and_design_masks(x)
        if target_mask is None or design_mask is None:
            return x
        if "hotspot" not in input_feature_dict:
            return x
        hotspot_tokens = input_feature_dict["hotspot"].to(device=device)
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"].to(device=device)
        hotspot_atom_mask = hotspot_tokens[..., atom_to_token_idx].to(torch.bool)
        hotspot_atom_mask = hotspot_atom_mask.reshape(
            *([1] * len(batch_shape)), 1, N_atom, 1
        )
        hotspot_atom_mask = hotspot_atom_mask.expand(*x.shape[:-1], 1) & target_mask
        framework_mask = _framework_mask_for_shape(x)
        cdr_mask = design_mask if framework_mask is None else design_mask & ~framework_mask
        return apply_hotspot_contact_guidance(
            x=x,
            cdr_mask=cdr_mask,
            hotspot_mask=hotspot_atom_mask,
            move_mask=design_mask,
            target_distance=hotspot_guidance_target_distance,
            weight=hotspot_guidance_weight,
        )

    def _apply_guidance(x, sigma, framework_noise):
        x = _apply_framework_guidance(x, sigma, framework_noise)
        x = _apply_clash_guidance(x)
        x = _apply_hotspot_guidance(x)
        return x

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training
        x_l = _initialize_condition_coords(x_l, noise_schedule[0])
        framework_noise = torch.randn(size=x_l.shape, device=device, dtype=dtype)
        x_l = _apply_guidance(x_l, noise_schedule[0], framework_noise)
        T = len(noise_schedule)
        for step_t, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )
            x_l = _apply_guidance(x_l, c_tau_last, framework_noise)

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )
            x_noisy = _apply_guidance(x_noisy, t_hat, framework_noise)

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
            )
            x_denoised = _apply_guidance(x_denoised, c_tau, framework_noise)

            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            if isinstance(step_scale_eta, float):
                eta = step_scale_eta
            elif step_scale_eta["type"] == "const":
                assert step_scale_eta["min"] == step_scale_eta["max"]
                eta = step_scale_eta["min"]
            else:
                eta_min, eta_max = step_scale_eta["min"], step_scale_eta["max"]
                if step_scale_eta["type"] == "linear":
                    eta = eta_min + (eta_max - eta_min) * (step_t / T)
                elif step_scale_eta["type"] == "poly":
                    eta = eta_min + (eta_max - eta_min) * (step_t / T) ** 2
                elif step_scale_eta["type"] == "cos":
                    eta = eta_min + 0.5 * (eta_max - eta_min) * (
                        1 - np.cos(np.pi * step_t / T)
                    )
                elif step_scale_eta["type"] == "piecewise":
                    eta = eta_min if step_t / T < 0.5 else eta_max
                elif step_scale_eta["type"] == "piecewise_65":
                    eta = eta_min if step_t / T < 0.65 else eta_max
                elif step_scale_eta["type"] == "piecewise_70":
                    eta = eta_min if step_t / T < 0.70 else eta_max
                else:
                    raise ValueError("Unsupported eta schedule!")
            x_l = x_noisy + eta * dt[..., None, None] * delta
            x_l = _apply_guidance(x_l, c_tau, framework_noise)

        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        print("diffusion_chunk_size: ", diffusion_chunk_size)
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)

        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l
