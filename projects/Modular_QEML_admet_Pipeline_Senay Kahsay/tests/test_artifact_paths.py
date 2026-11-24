"""Generated outputs must not mutate source trees or mirrored notebooks."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artifact_paths import PROTECTED_DIRECTORIES, validate_output_dir


class ArtifactPathTests(unittest.TestCase):
    def test_root_ancestors_and_protected_trees_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            forbidden = [root, root.parent, Path(root.anchor)]
            forbidden += [root / name / child for name in PROTECTED_DIRECTORIES for child in ("", "generated")]
            forbidden += [Path(tmp) / ".git" / "objects", root / "nested" / ".git" / "output"]
            for path in forbidden:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    validate_output_dir(path, project_root=root)

    def test_artifacts_and_external_outputs_are_allowed_without_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            for output in (root / "artifacts" / "validation", Path(tmp) / "external"):
                self.assertEqual(validate_output_dir(output, project_root=root), output.resolve())
                self.assertFalse(output.exists())

    def test_output_symlink_cannot_escape_into_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            (root / "src").mkdir(parents=True)
            alias = Path(tmp) / "alias"
            alias.symlink_to(root / "src", target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_output_dir(alias / "generated", project_root=root)

    def test_file_and_malformed_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            file = Path(tmp) / "file"
            file.write_text("fixture")
            for output in (file, None, True, ""):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    validate_output_dir(output, project_root=root)


if __name__ == "__main__":
    unittest.main()
