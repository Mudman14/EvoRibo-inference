import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io
import hashlib
import json

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import legalize, read_query, read_bpp, load_features
import download_model


class InputTests(unittest.TestCase):
    def test_projection(self):
        rng = np.random.default_rng(7)
        fi = rng.random((7, 5))
        fi /= fi.sum(-1, keepdims=True)
        fij = rng.random((7, 7, 25))
        a, b, metrics = legalize(fi, fij)
        self.assertEqual(b.shape, (7, 5, 7, 5))
        np.testing.assert_allclose(b, b.transpose(2, 3, 0, 1), atol=1e-6)
        for i in range(7):
            np.testing.assert_allclose(b[i, :, i, :], np.diag(a[i]), atol=1e-6)
            for j in range(7):
                np.testing.assert_allclose(b[i, :, j, :].sum(-1), a[i], atol=1e-4)
        self.assertLessEqual(metrics['marginal_max_abs_error'], 1e-4)

    def test_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = root/'query.fasta'
            query.write_text('>test\nACTG\n')
            self.assertEqual(read_query(query), 'ACUG')
            bpp = root/'bpp.prob'
            bpp.write_text('1 4 0.8\n')
            p = read_bpp(bpp, 4)
            self.assertEqual(p.shape, (4, 4))
            self.assertEqual(p[0, 3], p[3, 0])
            np.testing.assert_array_equal(p.diagonal(), 0)
            bpp.write_text('1 4 1.8\n')
            with self.assertRaises(ValueError):
                read_bpp(bpp, 4)
            feature = root/'features.npz'
            np.savez(feature, query='ACUG', embedding=np.zeros((4, 1280)), attention=np.zeros((4, 4, 660)))
            with self.assertRaises(ValueError):
                load_features(feature, 'AAAA')
            query.write_text('>one\nACUG\n>two\nACUG\n')
            with self.assertRaises(ValueError):
                read_query(query)

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            legalize(np.full((3, 5), np.nan), np.ones((3, 3, 25)))
        with self.assertRaises(ValueError):
            legalize(np.ones((3, 5))/5, np.ones((3, 5, 3, 5))/25)

    def test_channel_order(self):
        fi = np.array([[.1, .2, .3, .15, .25], [.25, .15, .1, .3, .2]])
        pairs = fi[:, None, :, None] * fi[None, :, None, :]
        _, fij, _ = legalize(fi, pairs.reshape(2, 2, 25))
        np.testing.assert_allclose(fij[0, :, 1, :], np.outer(fi[0], fi[1]), atol=1e-7)

    def test_download_failure_then_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'model.onnx'
            data = b'test artifact'
            manifest = json.dumps({'url': 'https://example.org/model',
                                   'sha256': hashlib.sha256(data).hexdigest()})
            with patch.object(sys, 'argv', ['download_model.py', '--out', str(out)]), \
                 patch.object(Path, 'read_text', return_value=manifest):
                with patch('urllib.request.urlopen', side_effect=OSError('Disconnected')):
                    with self.assertRaises(OSError):
                        download_model.main()
                self.assertEqual(list(Path(tmp).iterdir()), [])
                with patch('urllib.request.urlopen', return_value=io.BytesIO(data)):
                    download_model.main()
                self.assertEqual(out.read_bytes(), data)
                self.assertEqual(list(Path(tmp).iterdir()), [out])


if __name__ == '__main__':
    unittest.main()
