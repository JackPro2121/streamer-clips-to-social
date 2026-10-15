"""fetch_models selection semantics.

The render job fetches only ``transnetv2.onnx``; the SyncNet checkpoints belong
to the ML tier. A fetch of one model must therefore not fail because an
unrelated one is absent -- that bug stopped a publish run one minute in.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.fetch_models as fetch_models


def _entry(url: str, payload: bytes) -> dict:
    return {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "purpose": "test",
    }


class TestSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)
        self.payloads = {"a.onnx": b"AAA", "b.onnx": b"BBB"}
        self.registry = {
            name: _entry("http://example.invalid/" + name, payload)
            for name, payload in self.payloads.items()
        }
        patcher = mock.patch.object(fetch_models, "MODELS", self.registry)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def _fake_download(self, url: str, destination: Path, allow_missing: bool = False) -> bool:
        destination.write_bytes(self.payloads[Path(destination).name])
        return True

    def test_fetching_one_model_ignores_the_others(self) -> None:
        with mock.patch.object(fetch_models, "download", side_effect=self._fake_download) as download:
            code = fetch_models.process(self.models_dir, ["a.onnx"], False, False)
        self.assertEqual(code, 0)
        self.assertTrue((self.models_dir / "a.onnx").exists())
        self.assertFalse((self.models_dir / "b.onnx").exists())
        self.assertEqual(download.call_count, 1)

    def test_fetch_all_requires_everything(self) -> None:
        with mock.patch.object(fetch_models, "download", side_effect=self._fake_download):
            code = fetch_models.process(self.models_dir, ["all"], False, False)
        self.assertEqual(code, 0)
        self.assertTrue((self.models_dir / "b.onnx").exists())

    def test_verify_without_fetch_requires_everything(self) -> None:
        code = fetch_models.process(self.models_dir, [], True, False)
        self.assertEqual(code, 1)

    def test_present_but_wrong_hash_always_fails(self) -> None:
        (self.models_dir / "a.onnx").write_bytes(b"corrupt")
        with mock.patch.object(fetch_models, "download", side_effect=self._fake_download):
            code = fetch_models.process(self.models_dir, ["b.onnx"], False, False)
        self.assertEqual(code, 1)

    def test_unrequested_missing_model_does_not_fail(self) -> None:
        code = fetch_models.process(self.models_dir, ["a.onnx"], True, False)
        self.assertEqual(code, 1, "verify still requires the requested model")
        (self.models_dir / "a.onnx").write_bytes(self.payloads["a.onnx"])
        code = fetch_models.process(self.models_dir, ["a.onnx"], True, False)
        self.assertEqual(code, 0, "b.onnx missing must not fail a fetch scoped to a.onnx")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
