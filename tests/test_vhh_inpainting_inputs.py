from pathlib import Path

from pxdesign.utils.inputs import parse_yaml_to_json
from pxdesign.utils.vhh_inpainting import IMGT_HEAVY_CDR_RANGES, build_vhh_inpaint_plan


FRAMEWORK = Path(
    "../../external/RFantibody/scripts/examples/example_inputs/h-NbBCII10.pdb"
)


def test_build_vhh_inpaint_plan_masks_imgt_cdrs_and_changes_lengths():
    plan = build_vhh_inpaint_plan(
        framework_pdb=FRAMEWORK,
        framework_chain="H",
        cdr_lengths={"H1": 8, "H2": 7, "H3": 15},
    )

    assert IMGT_HEAVY_CDR_RANGES == {
        "H1": (27, 38),
        "H2": (56, 65),
        "H3": (105, 117),
    }
    assert len(plan.sequence) == 127 - 35 + 8 + 7 + 15
    assert plan.sequence.count("j") == 30
    assert plan.cdr_output_ranges == {
        "H1": (27, 34),
        "H2": (52, 58),
        "H3": (98, 112),
    }
    assert plan.fixed_source_resnums[1] == 1
    assert plan.fixed_source_resnums[35] == 39
    assert 27 not in plan.fixed_source_resnums


def test_parse_yaml_to_json_emits_vhh_inpainting_generation(tmp_path):
    yaml_path = tmp_path / "vhh.yaml"
    yaml_path.write_text(
        f"""
task_name: pdl1_vhh_inpaint
target:
  file: "{Path('examples/5o45.cif').resolve()}"
  chains:
    A:
      crop: ["1-116"]
      hotspots: [40, 99, 107]
vhh_framework:
  file: "{FRAMEWORK.resolve()}"
  chain: H
  numbering: imgt
  cdr_lengths:
    H1: 8
    H2: 7
    H3: 15
"""
    )

    tasks = parse_yaml_to_json(yaml_path)

    task = tasks[0]
    assert task["name"] == "pdl1_vhh_inpaint"
    assert "binder_length" not in task
    assert task["generation"][0]["type"] == "vhh_inpainting"
    assert task["generation"][0]["sequence"].count("j") == 30
    assert task["generation"][0]["framework_file"] == str(FRAMEWORK.resolve())
    assert task["generation"][0]["cdr_output_ranges"]["H3"] == [98, 112]
