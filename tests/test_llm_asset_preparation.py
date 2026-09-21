"""Focused offline tests for explicit Hugging Face asset preparation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fedbrew.data.llm_assets.config import (
    AssetConfigError,
    AssetPreparationConfig,
    load_asset_config,
)
from fedbrew.data.llm_assets.manifest import AssetManifestError, load_asset_manifest
from fedbrew.data.llm_assets.prepare import (
    AssetPreparationError,
    _has_verified_matching_cache,
    _load_pretrained,
    _resolved_revision,
)

pytestmark = pytest.mark.fast

try:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )
except ModuleNotFoundError:  # pragma: no cover - base install omits LLM extra.
    Tokenizer = None  # type: ignore[assignment,misc]


class LLMAssetPreparationTests(unittest.TestCase):
    def test_config_requires_explicit_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "assets.yaml"
            config_path.write_text(
                "model_identifier: local-model\n"
                "tokenizer_identifier: local-tokenizer\n"
                f"cache_dir: {Path(directory) / 'prepared'}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                AssetConfigError,
                "revision is required and must be non-empty",
            ):
                load_asset_config(config_path)

    def test_config_rejects_a_key_no_loader_reads(self) -> None:
        """A dropped key is worse than a missing one when it looks load-bearing.

        Three shipped configs carried ``trust_remote_code: false``. Nothing read
        it and nothing rejected it, so a line that reads as a security control
        asserted nothing at all.
        """

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "assets.yaml"
            config_path.write_text(
                "model_identifier: local-model\n"
                "tokenizer_identifier: local-tokenizer\n"
                "revision: " + "a" * 40 + "\n"
                f"cache_dir: {Path(directory) / 'prepared'}\n"
                "cach_dir: typo\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                AssetConfigError,
                "does not read: cach_dir",
            ):
                load_asset_config(config_path)

    def test_config_names_trust_remote_code_as_removed(self) -> None:
        """It is not configurable, and the error has to say why rather than
        list it as a stray key: preparation always loads with it False and the
        manifest reader refuses a manifest that says otherwise."""

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "assets.yaml"
            config_path.write_text(
                "model_identifier: local-model\n"
                "tokenizer_identifier: local-tokenizer\n"
                "revision: " + "a" * 40 + "\n"
                f"cache_dir: {Path(directory) / 'prepared'}\n"
                "trust_remote_code: false\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                AssetConfigError,
                "trust_remote_code has been removed",
            ):
                load_asset_config(config_path)

    def test_shipped_asset_configs_carry_only_keys_the_loader_reads(self) -> None:
        """The regression that motivated the allow-list, pinned against the
        files themselves rather than against a synthetic config."""

        config_root = Path(__file__).resolve().parent.parent / "configs" / "llm_assets"
        shipped = sorted(config_root.glob("*.yaml"))
        self.assertTrue(shipped, "no shipped LLM asset configs found")
        for path in shipped:
            with self.subTest(config=path.name):
                load_asset_config(path)

    def test_missing_manifest_names_exact_preparation_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = (
                Path(directory) / "data" / "cache" / "llm" / "tiny_gpt2" / "asset_manifest.json"
            )

            with self.assertRaisesRegex(
                AssetManifestError,
                "fedbrew prepare-llm --config configs/llm_assets/tiny_gpt2.yaml",
            ):
                load_asset_manifest(manifest_path)

    def test_remote_assets_require_a_resolved_commit(self) -> None:
        config = AssetPreparationConfig(
            model_identifier="example/model",
            tokenizer_identifier="example/model",
            revision="release-tag",
            cache_dir=Path("unused"),
            config_path=Path("assets.yaml"),
        )
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash=None))
        tokenizer = SimpleNamespace(init_kwargs={})

        with self.assertRaisesRegex(
            AssetPreparationError,
            "unable to determine the resolved commit revision for model",
        ):
            _resolved_revision(model, tokenizer, config)

    def _write_matching_manifest(
        self,
        directory: Path,
        *,
        revision: str = "a" * 40,
        resolved_revision: str = "a" * 40,
        model_identifier: str = "example/model",
    ) -> tuple[Path, AssetPreparationConfig]:
        """A hand-written, fully valid prepared-asset manifest plus its config.

        Mirrors what ``prepare_llm_assets`` itself writes -- ``model_path``/
        ``tokenizer_path`` directories exist, every manifest field is present
        -- without the cost of actually loading a model.
        """

        cache_root = directory / "prepared"
        (cache_root / "model").mkdir(parents=True)
        (cache_root / "tokenizer").mkdir(parents=True)
        manifest_path = cache_root / "asset_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_identifier": model_identifier,
                    "tokenizer_identifier": model_identifier,
                    "requested_revision": revision,
                    "resolved_revision": resolved_revision,
                    "cache_path": str(cache_root),
                    "model_path": "model",
                    "tokenizer_path": "tokenizer",
                    "vocabulary_size": 7,
                    "model_type": "gpt2",
                    "preparation_timestamp": "2026-01-01T00:00:00+00:00",
                    "trust_remote_code": False,
                }
            ),
            encoding="utf-8",
        )
        config = AssetPreparationConfig(
            model_identifier=model_identifier,
            tokenizer_identifier=model_identifier,
            revision=revision,
            cache_dir=cache_root,
            config_path=directory / "assets.yaml",
        )
        return manifest_path, config

    def test_verified_cache_matches_a_sha_pinned_prepared_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, config = self._write_matching_manifest(Path(directory))
            self.assertTrue(_has_verified_matching_cache(manifest_path, config))

    def test_verified_cache_requires_a_full_commit_sha(self) -> None:
        """A floating ref (e.g. a branch) can move on the Hub between runs.

        The only way to know it hasn't is to ask the Hub -- exactly the
        network round trip this shortcut exists to avoid -- so a non-SHA
        revision must never take the cache-hit path, even against a manifest
        that would otherwise satisfy every other check.
        """

        with tempfile.TemporaryDirectory() as directory:
            manifest_path, config = self._write_matching_manifest(
                Path(directory), revision="main", resolved_revision="a" * 40
            )
            self.assertFalse(_has_verified_matching_cache(manifest_path, config))

    def test_verified_cache_rejects_a_different_model_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, config = self._write_matching_manifest(Path(directory))
            config = AssetPreparationConfig(
                model_identifier="example/a-different-model",
                tokenizer_identifier=config.tokenizer_identifier,
                revision=config.revision,
                cache_dir=config.cache_dir,
                config_path=config.config_path,
            )
            self.assertFalse(_has_verified_matching_cache(manifest_path, config))

    def test_load_pretrained_failure_names_the_real_cause_not_the_revision(self) -> None:
        """A from_pretrained failure unrelated to the revision must say so.

        Regression: every exception from from_pretrained used to be relabeled
        "unable to resolve requested revision ... verify the identifier and
        pinned revision" regardless of cause. A real pinned revision
        (confirmed valid on the Hub) still hit that message when transformers
        refused torch.load on an old local torch -- the wrong diagnosis, 30+
        seconds after the call started. The wrapped message must now carry
        the underlying exception's own text.
        """

        config = AssetPreparationConfig(
            model_identifier="example/model",
            tokenizer_identifier="example/model",
            revision="a" * 40,
            cache_dir=Path("unused"),
            config_path=Path("assets.yaml"),
        )

        class _ExplodingFactory:
            @staticmethod
            def from_pretrained(identifier: str, **kwargs: object) -> None:
                raise ValueError("torch.load requires torch>=2.6 for this checkpoint format")

        with self.assertRaisesRegex(
            AssetPreparationError,
            "torch.load requires torch>=2.6",
        ) as raised:
            _load_pretrained(
                _ExplodingFactory,
                kind="model",
                identifier="example/model",
                config=config,
                kwargs={},
            )

        self.assertNotIn("verify the identifier and pinned revision", str(raised.exception))

    @unittest.skipIf(Tokenizer is None, "Transformers LLM dependencies unavailable")
    def test_local_fixture_prepares_self_contained_offline_assets(self) -> None:
        from fedbrew.data.llm_assets.prepare import STAGES, prepare_llm_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            prepared = root / "prepared"
            source.mkdir()
            self._write_local_hf_fixture(source)

            config_path = root / "assets.yaml"
            config_path.write_text(
                f"model_identifier: {source}\n"
                f"tokenizer_identifier: {source}\n"
                "revision: local-fixture-v1\n"
                f"cache_dir: {prepared}\n",
                encoding="utf-8",
            )

            import io

            from fedbrew.core.console import DONE, RAIL, build_surface

            buffer = io.StringIO()
            surface = build_surface(file=buffer)
            manifest_path = prepare_llm_assets(config_path, surface.rail(STAGES))
            manifest = load_asset_manifest(manifest_path)

            # The rail on the miss path: every stage settles, and each carries
            # a fact the command checked and used to discard. Asserted here
            # rather than in a separate module because this is the only test
            # that exercises a full preparation without the network.
            rendered = buffer.getvalue()
            self.assertIn(f"{RAIL} {DONE} cache", rendered)
            self.assertIn("no usable cache", rendered)
            self.assertIn(f"{RAIL} {DONE} model", rendered)
            self.assertIn("gpt2", rendered)
            self.assertIn(f"{RAIL} {DONE} tokenizer", rendered)
            self.assertIn("vocabulary 7", rendered)

            self.assertEqual(manifest.model_identifier, str(source))
            self.assertEqual(manifest.tokenizer_identifier, str(source))
            self.assertEqual(manifest.requested_revision, "local-fixture-v1")
            self.assertIsNone(manifest.resolved_revision)
            self.assertEqual(manifest.cache_path, str(prepared.resolve()))
            self.assertEqual(manifest.vocabulary_size, 7)
            self.assertEqual(manifest.model_type, "gpt2")
            self.assertFalse(manifest.trust_remote_code)
            self.assertTrue((manifest.model_asset_path / "config.json").is_file())
            self.assertTrue((manifest.tokenizer_asset_path / "tokenizer.json").is_file())

            offline = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            with patch.dict(os.environ, offline, clear=False):
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    manifest.model_asset_path,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                loaded_tokenizer = AutoTokenizer.from_pretrained(
                    manifest.tokenizer_asset_path,
                    local_files_only=True,
                    trust_remote_code=False,
                )

            self.assertEqual(loaded_model.config.vocab_size, 7)
            self.assertEqual(len(loaded_tokenizer), 7)
            serialized = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(serialized["model_path"], "model")
            self.assertEqual(serialized["tokenizer_path"], "tokenizer")

    @unittest.skipIf(Tokenizer is None, "Transformers LLM dependencies unavailable")
    def test_prepare_llm_assets_skips_a_verified_sha_pinned_cache(self) -> None:
        """A second prepare against an already-verified SHA-pinned cache must
        not reload the model/tokenizer at all -- prepare-llm had no cache-hit
        shortcut before this, unlike prepare-oasst1's, so every invocation
        re-downloaded/reloaded/resaved even against an asset already fully
        prepared. Patching _load_pretrained to raise proves the second call
        never reaches it.
        """

        from fedbrew.data.llm_assets.prepare import prepare_llm_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            prepared = root / "prepared"
            source.mkdir()
            self._write_local_hf_fixture(source)

            revision = "b" * 40
            config_path = root / "assets.yaml"
            config_path.write_text(
                f"model_identifier: {source}\n"
                f"tokenizer_identifier: {source}\n"
                f"revision: {revision}\n"
                f"cache_dir: {prepared}\n",
                encoding="utf-8",
            )

            first_manifest_path = prepare_llm_assets(config_path)
            first_manifest = load_asset_manifest(first_manifest_path)
            self.assertEqual(first_manifest.resolved_revision, revision)

            with patch(
                "fedbrew.data.llm_assets.prepare._load_pretrained",
                side_effect=AssertionError("cache hit should not reload the model/tokenizer"),
            ):
                second_manifest_path = prepare_llm_assets(config_path)

            self.assertEqual(second_manifest_path, first_manifest_path)
            second_manifest = load_asset_manifest(second_manifest_path)
            self.assertEqual(
                second_manifest.preparation_timestamp,
                first_manifest.preparation_timestamp,
            )

    def _write_local_hf_fixture(self, source: Path) -> None:
        vocabulary = {
            "<pad>": 0,
            "<eos>": 1,
            "<unk>": 2,
            "hello": 3,
            "federated": 4,
            "language": 5,
            "model": 6,
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            eos_token="<eos>",
            pad_token="<pad>",
        )
        tokenizer.save_pretrained(source)

        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocabulary),
                n_positions=16,
                n_ctx=16,
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=1,
                eos_token_id=1,
                pad_token_id=0,
            )
        )
        model.save_pretrained(source)


if __name__ == "__main__":
    unittest.main()
