"""All-atom LDDT metric using biotite."""

import logging
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, AtomArrayPlusStack, as_atom_array_plus_stack
from biotite.structure import AtomArray, AtomArrayStack, lddt

from rfproteina.metrics.metric import Metric
from rfproteina.utils.atom_permutation import permute_gt_atoms
from rfproteina.utils.subunit_permutation import permute_gt_subunits

logger = logging.getLogger(__name__)

AtomArrayInput: TypeAlias = (
    AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack | list[AtomArray | AtomArrayPlus]
)

SymmetryLevel: TypeAlias = Literal["atom", "subunit"]


def _get_occupancy_mask(atom_array: AtomArray) -> np.ndarray | None:
    """Get occupancy mask if available (occupancy > 0 means atom is present)."""
    if "occupancy" in atom_array.get_annotation_categories():
        return atom_array.occupancy > 0
    return None


def _to_plain_atom_array(arr: AtomArray | AtomArrayPlus) -> AtomArray:
    """Convert AtomArrayPlus to plain AtomArray for compatibility with some Biotite functions."""
    if isinstance(arr, AtomArrayPlus):
        return arr.as_atom_array()
    return arr


class AllAtomLDDT(Metric):
    """Computes all-atom LDDT scores using Biotite."""

    def __init__(
        self,
        predicted_atom_array_key: str = "predicted_atom_array",
        ground_truth_atom_array_key: str = "ground_truth_atom_array",
        inclusion_radius: float = 15.0,
        resolve_symmetry: Sequence[SymmetryLevel] = ("subunit", "atom"),
        max_subunit_permutations: int = 720,
        **kwargs,
    ):
        """Initialize the LDDT metric.

        Args:
            predicted_atom_array_key: Key for predicted AtomArray(s) in kwargs.
            ground_truth_atom_array_key: Key for ground truth AtomArray(s) in kwargs.
            inclusion_radius: Distance cutoff in Angstroms. Defaults to 15.0.
            resolve_symmetry: Enabled levels; defaults to subunit then atom. Empty disables both.
                Subunits are matched by aligned coordinate error before atom resolution and LDDT scoring.
            max_subunit_permutations: Maximum combined subunit assignments. Defaults to 720.

        Raises:
            ValueError: If `resolve_symmetry` names an unknown level.
            NotImplementedError: During scoring, if the subunit permutation limit is exceeded.
        """
        unknown = set(resolve_symmetry) - {"atom", "subunit"}
        if unknown:
            raise ValueError(f"Unknown symmetry level(s) {sorted(unknown)}; expected 'atom' and/or 'subunit'")

        # Set before super().__init__ because it accesses kwargs_to_compute_args
        self.predicted_atom_array_key = predicted_atom_array_key
        self.ground_truth_atom_array_key = ground_truth_atom_array_key
        self.inclusion_radius = inclusion_radius
        self.resolve_symmetry = tuple(resolve_symmetry)
        self.max_subunit_permutations = max_subunit_permutations
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted": self.predicted_atom_array_key,
            "ground_truth": self.ground_truth_atom_array_key,
        }

    def compute(
        self,
        predicted: AtomArrayInput,
        ground_truth: AtomArrayInput,
    ) -> dict[str, float]:
        """Compute all-atom LDDT using Biotite.

        Args:
            predicted: Predicted structure(s).
            ground_truth: Ground truth structure(s).

        Returns:
            Dictionary with 'all_atom_lddt' score.
        """
        if (isinstance(predicted, list) and not predicted) or (isinstance(ground_truth, list) and not ground_truth):
            return {"all_atom_lddt": float("nan")}
        pred_stack = as_atom_array_plus_stack(predicted)
        gt_stack = as_atom_array_plus_stack(ground_truth)
        n_predictions = pred_stack.stack_depth()
        if (
            not n_predictions
            or gt_stack.stack_depth() not in (1, n_predictions)
            or not pred_stack.array_length()
            or pred_stack.array_length() != gt_stack.array_length()
        ):
            logger.warning("lDDT needs matching atom counts and one shared reference or a reference per prediction.")
            return {"all_atom_lddt": float("nan")}
        if "subunit" in self.resolve_symmetry:
            gt_stack = as_atom_array_plus_stack(
                permute_gt_subunits(pred_stack, gt_stack, max_permutations=self.max_subunit_permutations)
            )
        if "atom" in self.resolve_symmetry:
            gt_stack = permute_gt_atoms(pred_stack, gt_stack)

        # Compute LDDT for each pair
        scores = []
        shared_reference = gt_stack.stack_depth() == 1
        for i in range(n_predictions):
            # Convert to plain AtomArray for compatibility with Biotite lddt function
            pred = _to_plain_atom_array(pred_stack[i])
            gt = _to_plain_atom_array(gt_stack[0 if shared_reference else i])
            # Use occupancy mask if available
            atom_mask = _get_occupancy_mask(gt)
            # ... and score
            score = lddt(gt, pred, atom_mask=atom_mask, inclusion_radius=self.inclusion_radius)
            scores.append(float(score))

        return {"all_atom_lddt": scores[0] if len(scores) == 1 else sum(scores) / len(scores)}
