"""Verified Gje interfaces and provenance only; no model/training computation.

PULSE keeps its historical run record. Extended models record their actual
main file, local implementation dependencies and checkpoint configuration.
"""

import hashlib
from pathlib import Path


GJE_SUPPORTED_MODELS = ('PULSE', 'PULSE_SCF', 'PULSE_FACT')


def gje_extended_model_provenance(model_name, model):
    """Describe an already-built extended model without changing its state."""
    if model_name not in ('PULSE_SCF', 'PULSE_FACT'):
        raise ValueError('No extended Gje provenance for ' + model_name)

    root = Path(__file__).resolve().parents[1]
    main_file = 'models/{}.py'.format(model_name)
    files = {
        main_file, 'layers/SelfAttention_Family.py',
        'run.py', 'exp/exp_main.py', 'exp/exp_basic.py',
        'data_provider/data_factory.py',
        'data_provider/data_loader_gjemnessund.py',
        'utils/gje_timefeatures.py', 'utils/gje_model_support.py',
        'utils/tools.py', 'utils/metrics.py', 'utils/masking.py',
    }
    namespace = 'pulse_scf' if model_name == 'PULSE_SCF' else 'pulse_fact'
    files.update(str(path.relative_to(root))
                 for path in (root / 'layers' / namespace).rglob('*.py'))
    if model_name == 'PULSE_SCF':
        files.add('models/PULSE.py')  # Inherited norm/SAM/Router/denorm.

    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in sorted(files)}
    return {
        'gje_model_provenance_version': 1,
        'model_name': model_name,
        'model_file': main_file,
        'model_sha256': hashes[main_file],
        'source_sha256': hashes,
        'effective_model_settings': model.get_extra_state(),
    }
