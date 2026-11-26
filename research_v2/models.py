"""Portable loading for the immutable v1 research checkpoints.

The registered CLI ran under ``python -m`` and joblib recorded two class names
under __main__. Map those exact names locally without changing __main__, old
artifacts, manifests, or the registered fitting source. Trusted local files only:
checksums establish integrity, not safety against malicious pickle authors.
"""
from pathlib import Path
import hashlib
import io
import json
import zlib
from joblib.numpy_pickle import NumpyUnpickler
from research_v2.experiments import Config, ControlledModel


class _CheckpointUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module == '__main__' and name in {'Config', 'ControlledModel'}:
            return {'Config': Config, 'ControlledModel': ControlledModel}[name]
        return super().find_class(module, name)


def load_checkpoint(path):
    """Load a checksummed compressed v1 model without relying on caller imports."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    blob = (path / 'model.joblib').read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest['sha256']:
        raise ValueError('Model checksum mismatch')
    # v1 explicitly registered joblib.dump(..., compress=3): zlib payload.
    payload = io.BytesIO(zlib.decompress(blob))
    obj = _CheckpointUnpickler(str(path / 'model.joblib'), payload, ensure_native_byte_order=True).load()
    if not isinstance(obj, ControlledModel) or obj.metadata != manifest['metadata']:
        raise ValueError('Model metadata/type mismatch')
    return obj
