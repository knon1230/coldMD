"""CIF loading and structural pre-flight checks.

The MD driver should never silently turn a crystallographic disorder model into
one arbitrary atomistic model.  ASE can preserve CIF occupancies in
``Atoms.info``, but an MD potential still needs one unambiguous element at every
site.  Consequently, fractional or unknown occupancies are rejected unless the
caller explicitly opts in.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from ase import Atoms
from ase.formula import Formula
from ase.io.cif import parse_cif


class StructureValidationError(ValueError):
    """Raised when an input structure is ambiguous or unsafe for MD."""


@dataclass(frozen=True, slots=True)
class StructureReport:
    """Summary of a successfully loaded and validated structure."""

    source: Path | None
    block_name: str | None
    block_index: int | None
    atom_count: int
    formula: str
    reduced_formula: str
    composition: dict[str, int]
    pbc: tuple[bool, bool, bool]
    cell_determinant_a3: float
    volume_a3: float
    minimum_distance_a: float | None
    fractional_occupancy_present: bool


BlockSelector = int | str | None
FormulaMode = Literal["exact", "reduced"]


def read_cif_structure(
    path: str | Path,
    *,
    block: BlockSelector = None,
    supercell: Sequence[int] = (1, 1, 1),
    allow_fractional_occupancy: bool = False,
    occupancy_tolerance: float = 1.0e-8,
    expected_atom_count: int | None = None,
    expected_formula: str | None = None,
    formula_mode: FormulaMode = "reduced",
    minimum_distance_a: float | None = 0.5,
    require_3d_pbc: bool = True,
    cell_determinant_tolerance_a3: float = 1.0e-10,
    reader: str = "ase",
) -> tuple[Atoms, StructureReport]:
    """Read one CIF data block and perform MD-oriented validation.

    Parameters
    ----------
    path
        CIF file to read.
    block
        Zero-based index among structure-bearing blocks (negative indices are
        accepted), or a CIF data-block name.  A leading ``data_`` in a name is
        optional.  If omitted, the file must contain exactly one
        structure-bearing block.
    supercell
        Three positive replication counts.  Atom-count and formula assertions
        are applied *after* this expansion.
    allow_fractional_occupancy
        Opt-in escape hatch.  ASE represents a disordered site by one primary
        chemical symbol plus occupancy metadata; it does *not* construct a
        physical occupational configuration.  Production MD should normally
        leave this ``False`` and prepare an ordered configuration separately.
    expected_formula
        Composition assertion.  ``formula_mode='reduced'`` compares elemental
        ratios (for example, ``SiO2`` accepts any stoichiometric supercell),
        while ``'exact'`` compares absolute atom counts.

    Returns
    -------
    (atoms, report)
        An ASE ``Atoms`` object and an immutable validation summary.
    """

    cif_path = Path(path).expanduser().resolve()
    if not cif_path.is_file():
        raise StructureValidationError(f"CIF file does not exist: {cif_path}")
    if occupancy_tolerance < 0.0:
        raise StructureValidationError("occupancy_tolerance must be non-negative")

    try:
        blocks = list(parse_cif(str(cif_path), reader=reader))
    except Exception as exc:  # ASE and optional pycodcif raise several types.
        raise StructureValidationError(
            f"Could not parse CIF {cif_path}: {type(exc).__name__}: {exc}"
        ) from exc

    if not blocks:
        raise StructureValidationError(f"CIF contains no data blocks: {cif_path}")

    selected, selected_index = _select_cif_block(blocks, block, cif_path)
    block_name = str(getattr(selected, "name", selected_index))
    fractional, occupancy_details = _fractional_occupancy_details(
        selected, tolerance=occupancy_tolerance
    )
    if fractional and not allow_fractional_occupancy:
        detail = "; ".join(occupancy_details[:8])
        if len(occupancy_details) > 8:
            detail += f"; ... ({len(occupancy_details) - 8} more)"
        raise StructureValidationError(
            "CIF block "
            f"{block_name!r} contains fractional or unknown site occupancies "
            f"({detail}). Classical/ML MD requires an explicit occupational "
            "configuration. Prepare an ordered model, or set "
            "allow_fractional_occupancy=true only if this ambiguity is "
            "intentional."
        )

    try:
        atoms = selected.get_atoms(
            store_tags=True,
            primitive_cell=False,
            subtrans_included=True,
            fractional_occupancies=True,
        )
    except Exception as exc:
        raise StructureValidationError(
            f"Could not construct atoms from CIF block {block_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    repeat = _validate_supercell(supercell)
    if repeat != (1, 1, 1):
        try:
            atoms = atoms.repeat(repeat)
        except Exception as exc:
            raise StructureValidationError(
                f"Could not expand CIF block {block_name!r} to supercell "
                f"{repeat}: {type(exc).__name__}: {exc}"
            ) from exc

    # Some ASE readers expose mixed occupancies only after Atoms construction.
    metadata_fractional, metadata_details = _atoms_occupancy_details(
        atoms, tolerance=occupancy_tolerance
    )
    fractional = fractional or metadata_fractional
    if metadata_fractional and not allow_fractional_occupancy:
        detail = "; ".join(metadata_details[:8])
        raise StructureValidationError(
            f"CIF block {block_name!r} contains mixed/fractional occupancy "
            f"metadata ({detail}). Prepare an explicit ordered configuration "
            "before MD."
        )

    report = validate_atoms(
        atoms,
        source=cif_path,
        block_name=block_name,
        block_index=selected_index,
        expected_atom_count=expected_atom_count,
        expected_formula=expected_formula,
        formula_mode=formula_mode,
        minimum_distance_a=minimum_distance_a,
        require_3d_pbc=require_3d_pbc,
        cell_determinant_tolerance_a3=cell_determinant_tolerance_a3,
        fractional_occupancy_present=fractional,
    )
    return atoms, report


# A concise alias for callers that currently accept CIF only.  Keeping the
# format-specific name above avoids pretending that occupancy checks generalize
# automatically to every ASE input format.
load_structure = read_cif_structure


def validate_atoms(
    atoms: Atoms,
    *,
    source: str | Path | None = None,
    block_name: str | None = None,
    block_index: int | None = None,
    expected_atom_count: int | None = None,
    expected_formula: str | None = None,
    formula_mode: FormulaMode = "reduced",
    minimum_distance_a: float | None = 0.5,
    require_3d_pbc: bool = True,
    cell_determinant_tolerance_a3: float = 1.0e-10,
    fractional_occupancy_present: bool = False,
) -> StructureReport:
    """Validate an already constructed ASE ``Atoms`` object.

    This function is also useful for unit tests and for structures produced by
    a preparation workflow rather than read directly from CIF.
    """

    if not isinstance(atoms, Atoms):
        raise StructureValidationError(
            f"Expected ase.Atoms, received {type(atoms).__name__}"
        )
    atom_count = len(atoms)
    if atom_count == 0:
        raise StructureValidationError("Input structure contains no atoms")
    if expected_atom_count is not None:
        if (
            isinstance(expected_atom_count, (bool, np.bool_))
            or not isinstance(expected_atom_count, (int, np.integer))
            or expected_atom_count <= 0
        ):
            raise StructureValidationError("expected_atom_count must be a positive integer")
        if atom_count != int(expected_atom_count):
            raise StructureValidationError(
                f"Atom-count mismatch: CIF has {atom_count}, expected "
                f"{expected_atom_count}"
            )

    positions = np.asarray(atoms.positions, dtype=float)
    if positions.shape != (atom_count, 3):
        raise StructureValidationError(
            f"Position array has shape {positions.shape}; expected ({atom_count}, 3)"
        )
    if not np.isfinite(positions).all():
        bad = np.argwhere(~np.isfinite(positions))[0]
        raise StructureValidationError(
            "Non-finite Cartesian coordinate at atom/axis "
            f"({int(bad[0])}, {int(bad[1])})"
        )

    numbers = np.asarray(atoms.numbers)
    if numbers.shape != (atom_count,) or np.any(numbers <= 0):
        raise StructureValidationError("Structure contains invalid atomic numbers")

    pbc = tuple(bool(value) for value in np.asarray(atoms.pbc, dtype=bool))
    if require_3d_pbc and pbc != (True, True, True):
        raise StructureValidationError(
            f"Cold-compression MD requires three-dimensional PBC; found pbc={pbc}"
        )

    cell = np.asarray(atoms.cell.array, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise StructureValidationError("Cell must be a finite 3 x 3 matrix")
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant):
        raise StructureValidationError("Cell determinant is non-finite")
    if cell_determinant_tolerance_a3 <= 0.0:
        raise StructureValidationError(
            "cell_determinant_tolerance_a3 must be positive"
        )
    if determinant <= cell_determinant_tolerance_a3:
        handedness = "left-handed/inverted" if determinant < 0.0 else "singular"
        raise StructureValidationError(
            f"Cell is {handedness}: det(cell)={determinant:.8g} A^3; "
            f"required > {cell_determinant_tolerance_a3:.3g} A^3"
        )

    composition = _composition(atoms)
    formula = atoms.get_chemical_formula(mode="hill")
    reduced_composition = _reduce_composition(composition)
    reduced_formula = _format_formula(reduced_composition)
    if expected_formula is not None:
        _assert_formula(
            actual=composition,
            expected=expected_formula,
            mode=formula_mode,
            actual_formula=formula,
            actual_reduced_formula=reduced_formula,
        )

    minimum = _minimum_periodic_distance(atoms)
    if minimum_distance_a is not None:
        if minimum_distance_a <= 0.0 or not np.isfinite(minimum_distance_a):
            raise StructureValidationError(
                "minimum_distance_a must be finite and positive, or None"
            )
        if minimum is not None and minimum < minimum_distance_a:
            raise StructureValidationError(
                f"Minimum interatomic distance is {minimum:.6f} A, below the "
                f"configured safety limit {minimum_distance_a:.6f} A. Check "
                "duplicate/symmetry-generated sites and CIF occupancies."
            )

    return StructureReport(
        source=Path(source).resolve() if source is not None else None,
        block_name=block_name,
        block_index=block_index,
        atom_count=atom_count,
        formula=formula,
        reduced_formula=reduced_formula,
        composition=composition,
        pbc=pbc,
        cell_determinant_a3=determinant,
        volume_a3=determinant,
        minimum_distance_a=minimum,
        fractional_occupancy_present=fractional_occupancy_present,
    )


def _select_cif_block(
    blocks: Sequence[Any], selector: BlockSelector, source: Path
) -> tuple[Any, int]:
    names = [str(getattr(item, "name", index)) for index, item in enumerate(blocks)]
    structure_indices = [
        index
        for index, item in enumerate(blocks)
        if _block_looks_structural(item)
    ]
    if isinstance(selector, bool):
        raise StructureValidationError("CIF block selector cannot be a boolean")
    if isinstance(selector, int):
        frame_index = (
            selector if selector >= 0 else len(structure_indices) + selector
        )
        if not 0 <= frame_index < len(structure_indices):
            raise StructureValidationError(
                f"CIF structure-frame index {selector} is out of range for "
                f"{len(structure_indices)} structure-bearing blocks: "
                f"{[(index, names[index]) for index in structure_indices]}"
            )
        index = structure_indices[frame_index]
        return blocks[index], index
    if isinstance(selector, str):
        wanted = _normalise_block_name(selector)
        matches = [
            index
            for index, name in enumerate(names)
            if _normalise_block_name(name) == wanted
        ]
        if not matches:
            raise StructureValidationError(
                f"CIF block {selector!r} was not found in {source}; available "
                f"blocks: {names}"
            )
        if len(matches) > 1:
            raise StructureValidationError(
                f"CIF block name {selector!r} is ambiguous; matching indices: {matches}"
            )
        return blocks[matches[0]], matches[0]
    if selector is not None:
        raise StructureValidationError(
            "CIF block selector must be an integer, string, or None"
        )

    if len(structure_indices) == 1:
        index = structure_indices[0]
        return blocks[index], index
    if not structure_indices:
        raise StructureValidationError(
            f"CIF contains no structure-bearing data block: {source}; blocks: {names}"
        )
    raise StructureValidationError(
        f"CIF contains {len(structure_indices)} structure-bearing blocks. "
        f"Select one by name or index: "
        f"{[(index, names[index]) for index in structure_indices]}"
    )


def _validate_supercell(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)):
        raise StructureValidationError(
            "supercell must contain three positive integers, not a string"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise StructureValidationError(
            "supercell must contain three positive integers"
        ) from exc
    if len(items) != 3:
        raise StructureValidationError(
            f"supercell must contain exactly three integers; received {items!r}"
        )
    if any(isinstance(item, (bool, np.bool_)) for item in items):
        raise StructureValidationError("supercell entries cannot be booleans")
    try:
        converted = tuple(int(item) for item in items)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StructureValidationError(
            f"supercell contains a non-integer value: {items!r}"
        ) from exc
    if any(converted[index] != items[index] for index in range(3)):
        raise StructureValidationError(
            f"supercell contains a non-integral value: {items!r}"
        )
    if any(item <= 0 for item in converted):
        raise StructureValidationError(
            f"supercell entries must be positive; received {converted!r}"
        )
    return converted  # type: ignore[return-value]


def _block_looks_structural(block: Any) -> bool:
    has_structure = getattr(block, "has_structure", None)
    if callable(has_structure):
        try:
            return bool(has_structure())
        except Exception:
            pass
    return any(
        _block_get(block, tag, None) is not None
        for tag in (
            "_atom_site_fract_x",
            "_atom_site_cartn_x",
            "_atom_site_type_symbol",
        )
    )


def _normalise_block_name(name: str) -> str:
    value = str(name).strip()
    if value.lower().startswith("data_"):
        value = value[5:]
    return value.casefold()


def _fractional_occupancy_details(
    block: Any, *, tolerance: float
) -> tuple[bool, list[str]]:
    raw = _block_get(block, "_atom_site_occupancy", None)
    if raw is None:
        return False, []
    values = _as_sequence(raw)
    labels = _as_sequence(_block_get(block, "_atom_site_label", []))
    details: list[str] = []
    for index, value in enumerate(values):
        label = str(labels[index]) if index < len(labels) else f"site {index}"
        try:
            occupancy = float(value)
        except (TypeError, ValueError):
            details.append(f"{label}: {value!r}")
            continue
        if not np.isfinite(occupancy) or abs(occupancy - 1.0) > tolerance:
            details.append(f"{label}: {occupancy:.8g}")
    return bool(details), details


def _atoms_occupancy_details(
    atoms: Atoms, *, tolerance: float
) -> tuple[bool, list[str]]:
    occupancy = atoms.info.get("occupancy")
    if not isinstance(occupancy, dict):
        return False, []
    details: list[str] = []
    for site, species_map in occupancy.items():
        if not isinstance(species_map, dict):
            details.append(f"site {site}: {species_map!r}")
            continue
        positive: list[tuple[str, float]] = []
        invalid = False
        for symbol, value in species_map.items():
            try:
                amount = float(value)
            except (TypeError, ValueError):
                invalid = True
                break
            if not np.isfinite(amount):
                invalid = True
                break
            if amount > tolerance:
                positive.append((str(symbol), amount))
        if invalid or len(positive) != 1 or abs(positive[0][1] - 1.0) > tolerance:
            details.append(f"site {site}: {species_map!r}")
    return bool(details), details


def _block_get(block: Any, key: str, default: Any) -> Any:
    getter = getattr(block, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return block[key]
    except (KeyError, TypeError):
        return default


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _composition(atoms: Atoms) -> dict[str, int]:
    result: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        result[symbol] = result.get(symbol, 0) + 1
    return result


def _reduce_composition(composition: dict[str, int]) -> dict[str, int]:
    divisor = reduce(gcd, composition.values())
    return {symbol: count // divisor for symbol, count in composition.items()}


def _format_formula(composition: dict[str, int]) -> str:
    symbols = list(composition)
    if "C" in composition:
        order = ["C"]
        if "H" in composition:
            order.append("H")
        order.extend(sorted(symbol for symbol in symbols if symbol not in order))
    else:
        order = sorted(symbols)
    return "".join(
        symbol + (str(composition[symbol]) if composition[symbol] != 1 else "")
        for symbol in order
    )


def _assert_formula(
    *,
    actual: dict[str, int],
    expected: str,
    mode: FormulaMode,
    actual_formula: str,
    actual_reduced_formula: str,
) -> None:
    if mode not in {"exact", "reduced"}:
        raise StructureValidationError(
            f"formula_mode must be 'exact' or 'reduced', received {mode!r}"
        )
    try:
        expected_counts = {
            symbol: int(count) for symbol, count in Formula(expected).count().items()
        }
    except Exception as exc:
        raise StructureValidationError(
            f"Invalid expected_formula {expected!r}: {exc}"
        ) from exc
    if not expected_counts or any(count <= 0 for count in expected_counts.values()):
        raise StructureValidationError(
            f"expected_formula must contain positive elemental counts: {expected!r}"
        )
    expected_for_comparison = (
        expected_counts if mode == "exact" else _reduce_composition(expected_counts)
    )
    actual_for_comparison = actual if mode == "exact" else _reduce_composition(actual)
    if actual_for_comparison != expected_for_comparison:
        raise StructureValidationError(
            f"Formula mismatch ({mode} comparison): structure is {actual_formula} "
            f"(reduced {actual_reduced_formula}), expected {expected!r}"
        )


def _minimum_periodic_distance(atoms: Atoms) -> float | None:
    if len(atoms) < 2:
        return None
    try:
        distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    except Exception as exc:
        raise StructureValidationError(
            f"Could not calculate periodic interatomic distances: {exc}"
        ) from exc
    np.fill_diagonal(distances, np.inf)
    minimum = float(np.min(distances))
    if not np.isfinite(minimum):
        raise StructureValidationError("Minimum interatomic distance is non-finite")
    return minimum


__all__ = [
    "BlockSelector",
    "FormulaMode",
    "StructureReport",
    "StructureValidationError",
    "load_structure",
    "read_cif_structure",
    "validate_atoms",
]
