"""Path layout checks independent of the plotting and Zarr dependencies."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for name, directory in (("shuga", ROOT), ("shuga.core", ROOT / "core")):
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules.setdefault(name, package)

for name in ("types", "naming", "paths"):
    spec = importlib.util.spec_from_file_location(f"shuga.core.{name}", ROOT / "core" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

from shuga.core.types import RunSpec
from shuga.core.paths import ShugaPaths


class PublicationPathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "afim_output"
        self.graphics = self.root / "GRAPHICAL"
        (self.out / "lateral-drag" / "Cs-high").mkdir(parents=True)
        (self.out / "LFI-waves-dyntens" / "snow-half").mkdir(parents=True)

    def paths(self, sim, publication=None, base=None):
        run = RunSpec(sim, "1995-01-01", "2005-12-31", publication=publication)
        return ShugaPaths(run_cfg=run, afim_output_root=base or self.out, graphics_root=self.graphics)

    def test_discovery_and_shared_static(self):
        paths = self.paths("Cs-high")
        self.assertEqual(paths.output_root, self.out / "lateral-drag" / "Cs-high")
        self.assertEqual(paths.figure_root(), self.graphics / "lateral-drag" / "Cs-high")
        self.assertEqual(paths.zarr_root, paths.output_root / "zarr")
        shared = self.out / "CICE_0p25_Cgrid_coords.zarr"
        shared.mkdir()
        self.assertEqual(paths.resolve_static_store(), shared)

    def test_explicit_publication_for_new_run_and_group_root(self):
        paths = self.paths("new-run", publication="LFI-waves-dyntens")
        self.assertEqual(paths.output_root, self.out / "LFI-waves-dyntens" / "new-run")
        group_paths = self.paths("new-run", base=self.out / "LFI-waves-dyntens")
        self.assertEqual(group_paths.output_root, paths.output_root)
        self.assertEqual(group_paths.figure_root(), self.graphics / "LFI-waves-dyntens" / "new-run")

    def test_ambiguous_and_missing_group(self):
        (self.out / "LFI-waves-dyntens" / "Cs-high").mkdir()
        with self.assertRaisesRegex(ValueError, "multiple publications"):
            _ = self.paths("Cs-high").output_root
        with self.assertRaisesRegex(ValueError, "set RunSpec"):
            _ = self.paths("unknown").output_root

    def test_existing_flat_layout(self):
        (self.out / "old-sim").mkdir()
        self.assertEqual(self.paths("old-sim").output_root, self.out / "old-sim")


if __name__ == "__main__":
    unittest.main()
