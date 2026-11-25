"""Minimal-core clean reproduction using exact archived fitting source.

Legacy src.models.__init__ eagerly imports unrelated optional training stacks.
This adapter installs only a package namespace and loads the exact archived
adaptive.py. No fitting mathematics, settings, partitions or data are changed.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import types
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/research-v2/experiments'
ARCHIVE = OUT / 'source'
design = json.loads((OUT / 'design.json').read_text())
for relative, expected in design['source_hashes'].items():
    for path in [ROOT / relative, ARCHIVE / relative]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Registered source mismatch: ' + str(path))
# Namespace-only package adapters bypass __init__, never the actual adaptive.py.
for name, path in [('src', ARCHIVE / 'src'), ('src.models', ARCHIVE / 'src/models'), ('research_v2', ARCHIVE / 'research_v2')]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package
source = ARCHIVE / 'research_v2/experiments.py'
spec = importlib.util.spec_from_file_location('research_v2.experiments', source)
module = importlib.util.module_from_spec(spec)
# Keep original repository path semantics for registered data/output lookup and
# source_hashes(). Both this path and the executed archive were checked above.
module.__file__ = str(ROOT / 'research_v2/experiments.py')
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if Path(sys.modules['src.models.adaptive'].__file__).resolve() != (ARCHIVE / 'src/models/adaptive.py').resolve():
    raise RuntimeError('Adaptive mathematical implementation did not load from archive')
os.environ.setdefault('EXPERIMENT_REPRODUCTION_OUTPUT', str(OUT / 'clean-reproduction.json'))
module.reproduce()
path = Path(os.environ['EXPERIMENT_REPRODUCTION_OUTPUT'])
report = json.loads(path.read_text())
report['compatibility_adapter'] = dict(
    source=str(Path(__file__).relative_to(ROOT)),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    executed_fitting_source=str(source.relative_to(ROOT)),
    executed_adaptive_source='data/research-v2/experiments/source/src/models/adaptive.py',
    registered_and_archived_sources_verified=True,
    namespace_only_packages=['src', 'src.models', 'research_v2'],
    bypassed_side_effect='src/models/__init__.py imports ModelTrainer and optional mlflow stack unrelated to the registered estimators',
    mathematics_changed=False,
)
path.write_text(json.dumps(report, indent=2) + '\n')
print('Actual archived-source clean reproduction passed: ' + str(path), flush=True)
