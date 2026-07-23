import json
import tempfile
import unittest
from pathlib import Path

from capture_contract import IMAGE_TO_INTRINSIC, POSES_FILE
from undistort import build_preprocess_cache_record, reusable_preprocess_manifest


class UndistortCacheTests(unittest.TestCase):
    def make_capture(self, root: Path) -> Path:
        capture = root / "capture"
        capture.mkdir()
        for index, (name, intrinsic) in enumerate(IMAGE_TO_INTRINSIC.items()):
            (capture / f"{name}.png").write_bytes(f"image-{index}".encode())
            (capture / intrinsic).write_text(
                json.dumps({"camera": name, "version": 1}), encoding="utf-8"
            )
        return capture

    def make_complete_output(self, root: Path, cache: dict) -> Path:
        output = root / "undistorted" / "capture"
        output.mkdir(parents=True)
        written = []
        for name in IMAGE_TO_INTRINSIC:
            (output / f"{name}.png").write_bytes(b"undistorted")
            (output / f"{name}_K.json").write_text("{}", encoding="utf-8")
            written.extend([f"{name}.png", f"{name}_K.json"])
        (output / "pipeline_preprocess_manifest.json").write_text(
            json.dumps(
                {"schema_version": 3, "cache": cache, "written_outputs": sorted(written)}
            ),
            encoding="utf-8",
        )
        return output

    def test_complete_matching_output_is_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self.make_capture(root)
            cache = build_preprocess_cache_record(capture)
            output = self.make_complete_output(root, cache)
            self.assertIsNotNone(reusable_preprocess_manifest(output, cache))

    def test_source_content_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self.make_capture(root)
            old_cache = build_preprocess_cache_record(capture)
            output = self.make_complete_output(root, old_cache)
            (capture / "head.png").write_bytes(b"changed-image")
            new_cache = build_preprocess_cache_record(capture)
            self.assertNotEqual(old_cache["key"], new_cache["key"])
            self.assertIsNone(reusable_preprocess_manifest(output, new_cache))

    def test_pose_mode_change_invalidates_cache_and_stale_pose_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self.make_capture(root)
            (capture / POSES_FILE).write_text('{"poses": {}}', encoding="utf-8")
            ignored_cache = build_preprocess_cache_record(capture, ignore_poses=True)
            output = self.make_complete_output(root, ignored_cache)
            (output / POSES_FILE).write_text("{}", encoding="utf-8")
            self.assertIsNone(reusable_preprocess_manifest(output, ignored_cache))
            metric_cache = build_preprocess_cache_record(capture, ignore_poses=False)
            self.assertNotEqual(ignored_cache["key"], metric_cache["key"])


if __name__ == "__main__":
    unittest.main()
