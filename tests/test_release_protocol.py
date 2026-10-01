import importlib.util
import os
import random
import tempfile
import unittest
import struct
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data_provider.data_loader import QARFlightDataset
from utils.benchmark_metrics import precision_at_recall
from utils.run_metadata import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RankingTests(unittest.TestCase):
    def test_first_recall_crossing_is_not_envelope(self):
        result = precision_at_recall([0, 1, 1, 1], [4, 3, 2, 1])
        self.assertEqual(result['precision'], 0.5)
        self.assertEqual(result['max_precision_at_recall_ge_target'], 0.75)

    def test_include_tied_scores(self):
        result = precision_at_recall([1, 0, 1], [0.9, 0.9, 0.1])
        self.assertEqual((result['tp'], result['fp']), (1, 1))

    def test_perfect_low_recall_does_not_mean_perfect_accuracy(self):
        result = precision_at_recall([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.1])
        self.assertEqual(result['precision'], 1.0)
        self.assertEqual(result['achieved_recall'], 0.5)

    def test_no_positive_is_undefined(self):
        self.assertIsNone(precision_at_recall([0, 0], [1, 2])['precision'])

    def test_invalid_inputs(self):
        for labels, scores in [([], []), ([1], [float('nan')]), ([2], [1]), ([0, 1], [1])]:
            with self.assertRaises(ValueError):
                precision_at_recall(labels, scores)


class ProtocolTests(unittest.TestCase):
    def test_seed_repeats_and_changes_initialization(self):
        def sample(seed):
            seed_everything(seed)
            return random.random(), np.random.rand(), torch.rand(3)
        a, b, c = sample(2021), sample(2021), sample(2022)
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertFalse(torch.equal(a[2], c[2]))

    def test_split_boundary_modes_are_explicit(self):
        with patch.dict(os.environ, QAR_SPLIT_BOUNDARY='integer'):
            self.assertEqual(QARFlightDataset._split_bounds_keep_test(10), (7, 8))
        with patch.dict(os.environ, QAR_SPLIT_BOUNDARY='legacy_float'):
            self.assertEqual(QARFlightDataset._split_bounds_keep_test(20), (14, 15))

    def test_dataset11_preprocessing_mode(self):
        prepare = load_script('prepare_release', 'tools/prepare/prepare_tsfile_compact_custom_conditions.py')
        self.assertEqual(prepare.classification_mode('dataset11'), 'standard_predict_2_3')
        self.assertEqual(prepare.classification_mode('dataset5'), 'standard_anchors')

    def test_rocket_split_variants(self):
        rocket = load_script('rocket_release', 'scripts/classification/run_qar_rocket_classifiers.py')
        for mode, expected in [('legacy_float', (14, 15)), ('rocket_float', (14, 16)),
                               ('integer', (14, 16)), ('floor_sum', (14, 16))]:
            with patch.dict(os.environ, QAR_SPLIT_BOUNDARY=mode):
                self.assertEqual(rocket.split_bounds_keep_test(20), expected)

    def test_history_shapes_and_normal_only(self):
        history = load_script('history_release', 'scripts/long_term_forecast/build_qar_history_forecast_compacts.py')
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'source.npz'
            np.savez_compressed(source, x=np.arange(8 * 80 * 2, dtype=np.float32).reshape(8, 80, 2),
                                labels=np.array([0, 1, 0, 1, 0, 0, 0, 0]),
                                sources=np.array([f'flight{i}' for i in range(8)]), time_keys=np.arange(8))
            for count in [2, 3, 5]:
                target = directory / f'hist{count}.npz'
                history.build_one(source, target, count, 80, 60, 20)
                with np.load(target) as cache:
                    self.assertEqual(cache['x'].shape, (8-count, count*80+80, 2))
                    self.assertEqual(int(cache['seq_len'][0]), count*80+60)
            normal = directory / 'normal.npz'
            history.build_one(source, normal, 2, 80, 60, 20, normal_only=True)
            with np.load(normal) as cache:
                self.assertTrue(np.all(cache['labels'] == 0))
                self.assertFalse(any('flight1' in s or 'flight3' in s for s in cache['history_sources']))
            before = normal.read_bytes()
            with self.assertRaises(ValueError):
                history.build_one(source, normal, 2, 80, 40, 20)
            self.assertEqual(normal.read_bytes(), before)

    def test_mamba_name_points_to_state_space_implementation(self):
        code = (ROOT / 'models/MambaSL.py').read_text()
        self.assertIn('from layers.MambaBlock import Mamba_TimeVariant', code)
        self.assertNotIn('class GatedTemporalBlock', code)
        self.assertFalse((ROOT / 'models/MambaSingleLayer.py').exists())


class DeidentificationMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools/prepare'))
        cls.module = load_script('deidentification_release', 'tools/prepare/deidentify_tsfile_archive.py')
        sys.path.pop(0)

    def test_new_zip64_offsets_are_allowed(self):
        for offset in [0, 2**31 + 123, 2**32 + 123]:
            info = zipfile.ZipInfo('sample.tsfile', date_time=(1980, 1, 1, 0, 0, 0))
            info.header_offset = offset
            if offset >= 2**31:
                info.extra = struct.pack('<HHQ', 1, 8, offset)
            self.assertTrue(self.module.neutral_zip_metadata(info))

    def test_original_timestamp_extra_is_rejected(self):
        info = zipfile.ZipInfo('sample.tsfile', date_time=(1980, 1, 1, 0, 0, 0))
        info.extra = struct.pack('<HHI', 0x5455, 4, 1234567890)
        self.assertFalse(self.module.neutral_zip_metadata(info))
        info.extra = b''
        info.comment = b'original flight metadata'
        self.assertFalse(self.module.neutral_zip_metadata(info))


if __name__ == '__main__':
    unittest.main()
