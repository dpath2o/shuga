"""Grouped model products and archive staging, without Gadi or model data."""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from shuga.core.types import ClassificationSpec, RunSpec
from shuga.core.paths import ShugaPaths


class ExperimentGroupPathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "afim_output"
        self.graphics = self.root / "GRAPHICAL"

    def paths(self, sim="new-run", group="lateral-drag", **kwargs):
        return ShugaPaths(run_cfg=RunSpec(sim, "1995-01-01", "2005-12-31", experiment_group=group),
                          cls_cfg=ClassificationSpec(),
                          afim_output_root=kwargs.pop("afim_output_root", self.out),
                          graphics_root=kwargs.pop("graphics_root", self.graphics), **kwargs)

    def test_defaults_for_all_three_new_experiments(self):
        for sim in ("free-slip-no-drag", "Cs-high-ktens-zero", "no-slip-LFI-match"):
            with self.subTest(sim=sim):
                p = ShugaPaths(run_cfg=RunSpec(sim, "1995-01-01", "2005-12-31"), cls_cfg=ClassificationSpec())
                expected = Path("/g/data/gv90/da1339/afim_output/lateral-drag") / sim
                self.assertEqual(p.output_root, expected)
                self.assertEqual(p.archive_root_path, expected)
                self.assertEqual(p.resolve_cice_store_target(), expected / "zarr/iceh_daily.zarr")
                self.assertEqual(p.classification_root_path, expected / "zarr/SH/ispd_thresh_5.0e-4/FI/Tc")
                self.assertEqual(p.figure_root(), Path("/g/data/gv90/da1339/GRAPHICAL/lateral-drag") / sim)

    def test_duplicate_names_do_not_trigger_cross_group_discovery(self):
        for group in ("lateral-drag", "paper3"):
            (self.out / group / "same").mkdir(parents=True)
        self.assertEqual(self.paths("same").output_root, self.out / "lateral-drag/same")
        self.assertEqual(self.paths("same", "paper3").output_root, self.out / "paper3/same")
        # An existing flat simulation cannot silently redirect new default writes.
        (self.out / "old-sim").mkdir()
        self.assertEqual(self.paths("old-sim").output_root, self.out / "lateral-drag/old-sim")

    def test_global_group_and_experiment_scoped_roots(self):
        for base in (self.out, self.out / "lateral-drag", self.out / "lateral-drag/new-run"):
            p = self.paths(afim_output_root=base)
            self.assertEqual(p.output_root, self.out / "lateral-drag/new-run")
            self.assertEqual(p.archive_root_path, p.output_root)
        # Explicit group-scoped roots remain supported for existing workflows.
        self.assertEqual(self.paths(afim_output_root=self.out / "paper1").experiment_group_name, "paper1")
        for base in (self.graphics, self.graphics / "lateral-drag", self.graphics / "lateral-drag/new-run"):
            self.assertEqual(self.paths(graphics_root=base).figure_root(), self.graphics / "lateral-drag/new-run")

    def test_legacy_archive_override_and_daily_hourly_readers(self):
        archive = self.root / "AFIM_archive"
        p = self.paths(archive_root=archive)
        for freq in ("daily", "hourly"):
            target = archive / "lateral-drag/new-run/history" / freq
            target.mkdir(parents=True)
            self.assertEqual(p.resolve_iceh_history_root(frequency=freq), target)
        store = p.archive_zarr_root_path / "iceh_daily.zarr"
        store.mkdir(parents=True)
        self.assertEqual(p.resolve_cice_store(), store)
        self.assertEqual(p.resolve_cice_store_target(), store)
        self.assertEqual(p.analysis_zarr_root_path, store.parent)

    def test_alias_validation_and_conflict(self):
        run = RunSpec("s", "1995-01-01", "2005-12-31", publication="paper3")
        self.assertEqual(run.experiment_group, "paper3")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            RunSpec("s", "1995-01-01", "2005-12-31", publication="paper1", experiment_group="paper3")
        for group in ("", "..", "a/b", "a\\b"):
            with self.subTest(group=group), self.assertRaises(ValueError):
                RunSpec("s", "1995-01-01", "2005-12-31", experiment_group=group)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _ = self.paths(group="paper3", afim_output_root=self.out / "paper1").output_root

    def test_switching_experiments_preserves_group_and_rebases_leaf_roots(self):
        p = self.paths("a", "paper3", afim_output_root=self.out / "paper3/a",
                       archive_root=self.root / "AFIM_archive/paper3/a",
                       graphics_root=self.graphics / "paper3/a",
                       cice_store=self.root / "a-only.zarr", classification_root=self.root / "a-only-class")
        other = p.for_sim("b")
        self.assertEqual(other.output_root, self.out / "paper3/b")
        self.assertEqual(other.archive_root_path, self.root / "AFIM_archive/paper3/b")
        self.assertEqual(other.figure_root(), self.graphics / "paper3/b")
        self.assertEqual(p.figure_root(sim_name="b"), other.figure_root())
        self.assertIsNone(other.cice_store)
        self.assertIsNone(other.classification_root)

    def test_shared_static_and_global_observations_are_not_grouped(self):
        p = self.paths()
        shared = self.out / "CICE_0p25_Cgrid_coords.zarr"
        shared.mkdir(parents=True)
        self.assertEqual(p.resolve_static_store(), shared)
        self.assertEqual(self.paths(afim_output_root=self.out / "lateral-drag").resolve_static_store(), shared)
        self.assertEqual(self.paths(afim_output_root=self.out / "lateral-drag/new-run").resolve_static_store(), shared)
        self.assertEqual(p.seaice_root_path, ShugaPaths().seaice_root_path)
        self.assertEqual(p.default_cice_static_store_path, Path("/g/data/gv90/da1339/afim_output/CICE_0p25_Cgrid_coords.zarr"))

    def test_loader_and_locator_keep_nondefault_group(self):
        from shuga.io.zarr_loading import _resolve_run_context, _build_paths
        from shuga.io.store_locator import CICEStoreLocator
        p = self.paths("a", "paper3")
        run, *_ = _resolve_run_context(run_cfg=p.run_cfg, sim_name="b")
        self.assertEqual(run.experiment_group, "paper3")
        self.assertEqual(_build_paths(run_cfg=run, pth_cfg=p).output_root, self.out / "paper3/b")
        locator = CICEStoreLocator(p.run_cfg, p.cls_cfg, pth_cfg=p)
        self.assertEqual(locator._paths_for_sim("b").output_root, self.out / "paper3/b")

    def test_public_loaders_read_selected_group_when_names_collide(self):
        import pandas as pd
        import xarray as xr
        from shuga.io.zarr_loading import load_metrics, load_classified
        for group, value in (("lateral-drag", 1.0), ("paper3", 2.0)):
            p = self.paths("b", group)
            for target, variable in ((p.metrics_store("binary-days"), "FIA"), (p.classification_store("binary-days"), "FI_mask")):
                target.parent.mkdir(parents=True, exist_ok=True)
                xr.Dataset({variable: ("time", [value])}, coords={"time": pd.to_datetime(["2000-01-01"])}).to_zarr(target, zarr_format=2)
        p = self.paths("a", "paper3")
        for loader, variable in ((load_metrics, "FIA"), (load_classified, "FI_mask")):
            ds = loader(run_cfg=p.run_cfg, pth_cfg=p, sim_name="b", classification="binary-days", variables=[variable])
            self.assertEqual(float(ds[variable].compute().item()), 2.0)
            ds.close()

    def test_metrics_wrapper_dry_run_passes_group_and_resolved_store(self):
        store = self.out / "paper3/new-run/zarr/iceh_daily.zarr"
        store.mkdir(parents=True)
        static = self.root / "static.zarr"
        static.mkdir()
        result = subprocess.run(["bash", str(ROOT / "scripts/metrics/metrics_pbs_wrapper.sh"),
                                 "-s", "new-run", "--experiment-group", "paper3",
                                 "--afim-output-root", str(self.out), "--static-store", str(static), "--dry-run"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(store), result.stdout)
        self.assertIn("PUBLICATION=paper3", result.stdout)


class ArchiveStagingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"
        self.archive = self.root / "afim_output"
        self.mapping = {"free-slip": "free-slip-no-drag", "free-slip01": "Cs-high-ktens-zero", "free-slip02": "no-slip-LFI-match"}
        for case in self.mapping:
            d = self.runs / case
            (d / "history").mkdir(parents=True)
            (d / "restart").mkdir()
            (d / "history/iceh.1995-01-01.nc").write_bytes(b"synthetic history")
            (d / "ice_in").write_text("namelist")
            (d / "restart/iced.1995-02-01.nc").write_bytes(b"synthetic restart")

    def call(self, *extra):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/admin/archive.py"), "--runs-root", str(self.runs), "--archive-root", str(self.archive), *extra], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_dry_run_does_not_create_archive(self):
        result = self.call()
        self.assertIn("lateral-drag/free-slip-no-drag", result.stdout)
        self.assertFalse(self.archive.exists())

    def test_new_destinations_stage_all_three_cases_without_changing_sources(self):
        self.call("--send-to-archive")
        for case, exp in self.mapping.items():
            dest = self.archive / "lateral-drag" / exp
            self.assertEqual((dest / "history/daily/iceh.1995-01-01.nc").read_bytes(), b"synthetic history")
            self.assertTrue((dest / "run_metadata" / case / "ice_in").is_file())
            self.assertTrue((dest / "restart/last" / case / "iced.1995-02-01.nc").is_file())
            self.assertTrue((self.runs / case / "history/iceh.1995-01-01.nc").is_file())
            self.assertFalse((self.archive / exp).exists())

    def test_both_month_spellings_skip_processed_history(self):
        for case, exp in self.mapping.items():
            month = "1995_01" if case == "free-slip01" else "1995-01"
            (self.archive / "lateral-drag" / exp / "zarr/iceh_daily.zarr" / month).mkdir(parents=True)
        self.call("--send-to-archive")
        for exp in self.mapping.values():
            self.assertFalse((self.archive / "lateral-drag" / exp / "history").exists())

    def test_usage_report_discovers_grouped_metadata(self):
        self.call("--send-to-archive")
        spec = importlib.util.spec_from_file_location("pbs_usage", ROOT / "scripts/admin/cice_pbs_usage.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.detect_layout(self.archive), "archive-metadata")
        records = mod.find_archive_metadata_dirs(self.archive)
        self.assertEqual({rec["experiment"] for rec in records}, {"lateral-drag/" + exp for exp in self.mapping.values()})

    def test_custom_group_and_already_scoped_root(self):
        self.call("--experiment-group", "paper3", "--send-to-archive")
        self.assertTrue((self.archive / "paper3/free-slip-no-drag/history/daily").is_dir())
        self.archive = self.archive / "paper3"
        self.call("--publication", "paper3", "--send-to-archive")
        self.assertFalse((self.archive / "paper3").exists())


if __name__ == "__main__":
    unittest.main()
