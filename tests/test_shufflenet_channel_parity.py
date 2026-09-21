"""An odd ShuffleNet stage width must be refused, not silently narrowed.

A ShuffleNet-V2 unit splits into two branches of ``output_channels // 2`` and
concatenates them, so an odd width produces ``output_channels - 1`` channels.
Nothing downstream notices immediately: the next unit is constructed from the
*requested* width, so the mismatch surfaces as a shape error one or more layers
later, and for the final stage not until the head. ``stage_channels`` is a
user-settable config key, which makes this reachable from a config typo.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.models.openimage_shufflenet import (
    ShuffleUnit,
    build_openimage_shufflenet,
)

pytestmark = pytest.mark.fast


class ShuffleUnitParityTest(unittest.TestCase):
    def test_an_odd_width_is_refused_at_construction(self) -> None:
        for stride in (1, 2):
            with self.subTest(stride=stride):
                with self.assertRaises(ValueError) as caught:
                    ShuffleUnit(115, 115, stride, 8)
                self.assertIn("must be even", str(caught.exception))

    def test_the_error_names_the_config_key_and_the_value(self) -> None:
        """A message that does not name either leaves the user guessing."""

        with self.assertRaises(ValueError) as caught:
            ShuffleUnit(24, 115, 2, 8)
        message = str(caught.exception)
        self.assertIn("stage_channels", message)
        self.assertIn("115", message)
        # The width it would otherwise have produced, so the report matches
        # what a shape error downstream would have shown.
        self.assertIn("114", message)

    def test_even_widths_still_construct_and_keep_their_width(self) -> None:
        """The guard must not reject anything that previously worked."""

        for stride in (1, 2):
            for channels in (2, 24, 116, 232, 464):
                with self.subTest(stride=stride, channels=channels):
                    unit = ShuffleUnit(channels, channels, stride, 8).eval()
                    with torch.no_grad():
                        out = unit(torch.zeros(1, channels, 8, 8))
                    self.assertEqual(out.shape[1], channels)


class ShuffleNetConfigParityTest(unittest.TestCase):
    def test_an_odd_stage_channel_is_refused_through_the_builder(self) -> None:
        """The reachable path: a config typo in model.stage_channels."""

        with self.assertRaises(ValueError) as caught:
            build_openimage_shufflenet({"stage_channels": [116, 231, 464]})
        self.assertIn("231", str(caught.exception))

    def test_the_shipped_defaults_are_unaffected(self) -> None:
        model = build_openimage_shufflenet({}).eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 596))


if __name__ == "__main__":
    unittest.main()
