"""`group_norm_groups: 8` does not mean 8 groups, and the run must say so.

`nn.GroupNorm` needs ``num_channels % num_groups == 0``, so one config number
cannot serve widths that are not all multiples of it. Both vision models walk
the request down to a divisor. On `femnist_resnet18` that never bites; on
`openimage_shufflenet` it bites over most of the network -- 13 layers at 2
groups and 26 at 4, against a config, a `run.json` and a sweep label that all
said 8. The backoff itself is deliberate (refusing would make 8 illegal for
that model, since only 1 and 2 divide every one of its widths), so what this
pins is that the reduction is reported rather than swallowed. P10-F32.
"""

from __future__ import annotations

import unittest

import pytest
from torch import nn

from fedbrew.models.femnist_resnet import build_femnist_resnet18
from fedbrew.models.group_norm import (
    REQUESTED_ATTRIBUTE,
    group_norm,
    group_norm_reductions,
    honoured_groups,
)
from fedbrew.models.openimage_shufflenet import build_openimage_shufflenet
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

pytestmark = pytest.mark.fast

#: configs/openimage/fedavg.yaml, less the keys the factory injects.
SHIPPED_OPENIMAGE = {
    "input_channels": 3,
    "num_classes": 596,
    "stage_channels": [116, 232, 464],
    "blocks_per_stage": [4, 8, 4],
    "stem_channels": 24,
    "final_channels": 1024,
    "group_norm_groups": 8,
    "stem_pool": False,
    "dropout": 0.1,
}

#: configs/femnist/fedavg.yaml, same.
SHIPPED_FEMNIST = {
    "input_channels": 1,
    "base_channels": 32,
    "blocks_per_stage": [2, 2, 2, 2],
    "group_norm_groups": 8,
    "dropout": 0.1,
    "num_classes": 62,
}

#: What the shipped OpenImage config actually builds, measured. Also the table
#: in the model's module docstring and in docs/06 section 4.3.
SHIPPED_OPENIMAGE_REDUCTIONS = [
    {"channels": 58, "requested": 8, "groups": 2, "layers": 13},
    {"channels": 116, "requested": 8, "groups": 4, "layers": 26},
]


class TheArithmeticTest(unittest.TestCase):
    def test_a_width_that_divides_gets_what_it_asked_for(self) -> None:
        for channels in (24, 232, 1024, 32, 64, 128, 256):
            with self.subTest(channels=channels):
                self.assertEqual(honoured_groups(channels, 8), 8)

    def test_a_width_that_does_not_divide_walks_down_to_the_nearest_divisor(self) -> None:
        # Not to 1, and not up: 58 = 2 * 29 and 116 = 4 * 29.
        self.assertEqual(honoured_groups(58, 8), 2)
        self.assertEqual(honoured_groups(116, 8), 4)

    def test_a_request_wider_than_the_tensor_is_capped(self) -> None:
        self.assertEqual(honoured_groups(4, 8), 4)

    def test_a_non_positive_request_is_refused_rather_than_dividing_by_zero(self) -> None:
        """Before, `group_norm_groups: 0` reached `channels % 0`."""

        for requested in (0, -4):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError) as caught:
                    honoured_groups(58, requested)
                self.assertIn("group_norm_groups must be positive", str(caught.exception))


class TheBuiltLayerRemembersTheRequestTest(unittest.TestCase):
    def test_the_module_carries_the_count_that_was_asked_for(self) -> None:
        norm = group_norm(58, 8)
        self.assertEqual(norm.num_groups, 2)
        self.assertEqual(getattr(norm, REQUESTED_ATTRIBUTE), 8)

    def test_the_request_is_not_part_of_the_state_that_gets_aggregated(self) -> None:
        """A tag in `state_dict` would be averaged across clients."""

        self.assertEqual(sorted(group_norm(58, 8).state_dict()), ["bias", "weight"])

    def test_a_group_norm_built_by_hand_reports_nothing(self) -> None:
        """No request was recorded, so there is no disagreement to report."""

        self.assertEqual(group_norm_reductions(nn.GroupNorm(2, 58)), [])


class TheShippedModelsReportWhatTheyBuiltTest(unittest.TestCase):
    def test_the_shipped_shufflenet_reduces_exactly_here(self) -> None:
        model = build_openimage_shufflenet(SHIPPED_OPENIMAGE)
        self.assertEqual(group_norm_reductions(model), SHIPPED_OPENIMAGE_REDUCTIONS)

    def test_every_other_shufflenet_width_did_get_eight(self) -> None:
        """So the report names the reduced widths and only those."""

        model = build_openimage_shufflenet(SHIPPED_OPENIMAGE)
        built: dict[int, set[int]] = {}
        for module in model.modules():
            if isinstance(module, nn.GroupNorm):
                built.setdefault(module.num_channels, set()).add(module.num_groups)
        self.assertEqual(built, {24: {8}, 58: {2}, 116: {4}, 232: {8}, 1024: {8}})

    def test_the_shipped_femnist_resnet_is_honoured_at_every_width(self) -> None:
        """A reporter that always reports would fail here."""

        model = build_femnist_resnet18(SHIPPED_FEMNIST)
        self.assertEqual(group_norm_reductions(model), [])
        self.assertTrue(any(isinstance(module, nn.GroupNorm) for module in model.modules()))


class TheRunRecordCarriesItTest(unittest.TestCase):
    def _metadata(self, model: nn.Module) -> dict:
        return TorchClassificationTask(device="cpu").federated_model_state_metadata(model)

    def test_run_json_records_the_reduction_beside_the_parameter_counts(self) -> None:
        metadata = self._metadata(build_openimage_shufflenet(SHIPPED_OPENIMAGE))
        self.assertEqual(metadata["group_norm_reductions"], SHIPPED_OPENIMAGE_REDUCTIONS)
        # The dict it joins is still the communication contract it was.
        self.assertEqual(metadata["model_state_scope"], "full")
        self.assertEqual(metadata["total_parameters"], 1864504)

    def test_an_honoured_request_leaves_no_entry_at_all(self) -> None:
        """An empty list reads as a measurement of nothing; absence is right."""

        metadata = self._metadata(build_femnist_resnet18(SHIPPED_FEMNIST))
        self.assertNotIn("group_norm_reductions", metadata)


if __name__ == "__main__":
    unittest.main()
