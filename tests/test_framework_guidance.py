import torch

from pxdesign.model.generator import (
    align_reference_to_masked_coords,
    apply_hotspot_contact_guidance,
    apply_interchain_clash_guidance,
    apply_masked_guidance,
    initialize_masked_coords,
)


def test_apply_masked_guidance_softly_moves_only_masked_atoms():
    x = torch.zeros(1, 2, 3)
    reference = torch.full_like(x, 10.0)
    mask = torch.tensor([[[True], [False]]])

    guided = apply_masked_guidance(x, reference, mask, weight=0.25)

    assert torch.allclose(guided[:, 0], torch.full((1, 3), 2.5))
    assert torch.allclose(guided[:, 1], torch.zeros(1, 3))


def test_apply_masked_guidance_noops_when_weight_is_zero():
    x = torch.randn(2, 3, 3)
    reference = torch.zeros_like(x)
    mask = torch.ones(2, 3, 1, dtype=torch.bool)

    guided = apply_masked_guidance(x, reference, mask, weight=0.0)

    assert torch.equal(guided, x)


def test_align_reference_to_masked_coords_preserves_free_rigid_body_pose():
    reference = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    coords = torch.tensor(
        [[[10.0, -2.0, 3.0], [10.0, -1.0, 3.0], [9.0, -2.0, 3.0]]]
    )
    mask = torch.ones(1, 3, 1, dtype=torch.bool)

    aligned = align_reference_to_masked_coords(reference, coords, mask)

    assert torch.allclose(aligned, coords, atol=1e-5)


def test_apply_interchain_clash_guidance_rigidly_moves_design_chain_outward():
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    target_mask = torch.tensor([[[True], [False], [False]]])
    design_mask = torch.tensor([[[False], [True], [True]]])

    guided = apply_interchain_clash_guidance(
        x=x,
        target_mask=target_mask,
        design_mask=design_mask,
        threshold=2.0,
        weight=1.0,
    )

    assert torch.allclose(guided[:, 0], x[:, 0])
    assert torch.allclose(guided[:, 1], torch.tensor([[2.0, 0.0, 0.0]]))
    assert torch.allclose(guided[:, 2], torch.tensor([[5.0, 0.0, 0.0]]))


def test_initialize_masked_coords_uses_reference_plus_sigma_noise():
    x = torch.zeros(1, 2, 3)
    reference = torch.full_like(x, 10.0)
    noise = torch.full_like(x, 2.0)
    mask = torch.tensor([[[True], [False]]])

    initialized = initialize_masked_coords(
        x=x,
        reference=reference,
        mask=mask,
        sigma=0.5,
        noise=noise,
    )

    assert torch.allclose(initialized[:, 0], torch.full((1, 3), 11.0))
    assert torch.allclose(initialized[:, 1], torch.zeros(1, 3))


def test_apply_hotspot_contact_guidance_rigidly_moves_vhh_toward_hotspot():
    x = torch.tensor([[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [25.0, 0.0, 0.0]]])
    hotspot_mask = torch.tensor([[[True], [False], [False]]])
    cdr_mask = torch.tensor([[[False], [True], [False]]])
    move_mask = torch.tensor([[[False], [True], [True]]])

    guided = apply_hotspot_contact_guidance(
        x=x,
        cdr_mask=cdr_mask,
        hotspot_mask=hotspot_mask,
        move_mask=move_mask,
        target_distance=10.0,
        weight=1.0,
    )

    assert torch.allclose(guided[:, 0], x[:, 0])
    assert torch.allclose(guided[:, 1], torch.tensor([[10.0, 0.0, 0.0]]))
    assert torch.allclose(guided[:, 2], torch.tensor([[15.0, 0.0, 0.0]]))
