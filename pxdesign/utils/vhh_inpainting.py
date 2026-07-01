"""VHH framework inpainting helpers using IMGT heavy-chain CDR ranges."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


IMGT_HEAVY_CDR_RANGES: Dict[str, Tuple[int, int]] = {
    "H1": (27, 38),
    "H2": (56, 65),
    "H3": (105, 117),
}

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class VHHInpaintPlan:
    sequence: str
    cdr_output_ranges: Dict[str, Tuple[int, int]]
    fixed_source_resnums: Dict[int, int]


def build_vhh_inpaint_plan(
    *,
    framework_pdb: Path,
    framework_chain: str,
    cdr_lengths: Mapping[str, int],
    cdr_ranges: Mapping[str, Tuple[int, int]] = IMGT_HEAVY_CDR_RANGES,
) -> VHHInpaintPlan:
    """Return an output VHH sequence with IMGT CDRs replaced by ``j`` masks."""
    residues = _read_chain_ca_sequence(Path(framework_pdb), framework_chain)
    if not residues:
        raise ValueError(
            f"No CA residues found for chain {framework_chain!r} in {framework_pdb}"
        )

    ordered_cdrs = sorted(cdr_ranges.items(), key=lambda item: item[1][0])
    sequence: List[str] = []
    fixed_source_resnums: Dict[int, int] = {}
    cdr_output_ranges: Dict[str, Tuple[int, int]] = {}
    output_pos = 1

    for resnum, aa in residues:
        cdr_name = _cdr_name_for_resnum(resnum, ordered_cdrs)
        if cdr_name is None:
            sequence.append(aa)
            fixed_source_resnums[output_pos] = resnum
            output_pos += 1
            continue

        start, _end = cdr_ranges[cdr_name]
        if resnum != start:
            continue
        length = int(cdr_lengths.get(cdr_name, cdr_ranges[cdr_name][1] - start + 1))
        cdr_start = output_pos
        sequence.extend("j" for _ in range(length))
        output_pos += length
        cdr_output_ranges[cdr_name] = (cdr_start, output_pos - 1)

    missing = set(cdr_ranges) - set(cdr_output_ranges)
    if missing:
        raise ValueError(f"Framework is missing IMGT CDR ranges: {sorted(missing)}")

    return VHHInpaintPlan(
        sequence="".join(sequence),
        cdr_output_ranges=cdr_output_ranges,
        fixed_source_resnums=fixed_source_resnums,
    )


def _read_chain_ca_sequence(path: Path, chain: str) -> List[Tuple[int, str]]:
    residues: List[Tuple[int, str]] = []
    seen = set()
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain or line[12:16].strip() != "CA":
                continue
            resnum = int(line[22:26])
            if resnum in seen:
                continue
            seen.add(resnum)
            residues.append((resnum, THREE_TO_ONE.get(line[17:20].strip(), "X")))
    return residues


def _cdr_name_for_resnum(
    resnum: int,
    ordered_cdrs: List[Tuple[str, Tuple[int, int]]],
) -> str | None:
    for name, (start, end) in ordered_cdrs:
        if start <= resnum <= end:
            return name
    return None
