"""FACT-specific integration; never changes legacy argument defaults or models."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


FACT_DEFAULTS = dict(d_model=512, d_ff=2048, core=0.5,
                     dilation=[1, 2, 3, 2, 1], num_kernels=4,
                     use_norm=1, dropout=0.1)


def add_fact_arguments(parser):
    """Absent options add no fields, preserving all old Gje config digests."""
    for name in ('d_model', 'd_ff', 'num_kernels', 'use_norm'):
        parser.add_argument('--fact_' + name, type=int, default=argparse.SUPPRESS)
    for name in ('core', 'dropout'):
        parser.add_argument('--fact_' + name, type=float, default=argparse.SUPPRESS)
    parser.add_argument('--fact_dilation', type=int, nargs='+', default=argparse.SUPPRESS)


def fact_model_config(args):
    values = {key: getattr(args, 'fact_' + key, value) for key, value in FACT_DEFAULTS.items()}
    for name in ('d_model', 'd_ff', 'num_kernels'):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('fact_' + name + ' must be a positive integer')
    if not 0 < values['core'] <= 1:
        raise ValueError('fact_core must be in (0,1]; original core=0 forecast has a known conv_time bug')
    if values['core'] != 1 and values['d_model'] % 2:
        raise ValueError('Original FACT frequency branch requires an even fact_d_model')
    if values['use_norm'] not in (0, 1) or not 0 <= values['dropout'] < 1:
        raise ValueError('Invalid fact_use_norm or fact_dropout')
    if not values['dilation'] or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                                     for v in values['dilation']):
        raise ValueError('fact_dilation must contain positive integers')
    if min(args.seq_len, args.pred_len, args.enc_in) < 1:
        raise ValueError('FACT sequence lengths and enc_in must be positive')
    grid = (math.isqrt(args.enc_in) + 1) ** 2
    if grid - args.enc_in > args.enc_in:
        raise ValueError('Original FACT single-prefix padding cannot fill this variable grid')
    return SimpleNamespace(task_name='long_term_forecast', seq_len=args.seq_len,
                           pred_len=args.pred_len, enc_in=args.enc_in, freq='n', **values)


def validate_fact_arguments(args):
    fact_model_config(args)
    if args.features != 'M' or args.label_len != 0:
        raise ValueError('FACT baseline supports --features M --label_len 0')
    if args.use_amp or args.use_multi_gpu or args.do_predict or args.output_attention:
        raise ValueError('FACT baseline supports non-AMP, single-device train/test only')
    if args.add_noise or args.loss.lower() != 'mse' or args.rec_lambda != 1 or args.auxi_lambda != 0:
        raise ValueError('FACT baseline requires --loss mse --rec_lambda 1 --auxi_lambda 0 and no noise')
    if (min(args.batch_size, args.train_epochs, args.patience) < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Invalid FACT training budget')
    if args.data == 'Gjemnessund':
        if args.enc_in != 10 or args.time_dim != 6:
            raise ValueError('FACT Gje requires --enc_in 10 --time_dim 6')
        if min(args.gje_record_limit, args.gje_max_batches) < 0 or min(
                args.gje_window_stride, args.gje_cycle_len, args.gje_log_interval) < 1:
            raise ValueError('Invalid Gje adapter/debug arguments')


def fact_setting(setting, args):
    if args.model != 'FACT':
        return setting
    config = vars(fact_model_config(args))
    budget = {name: getattr(args, name) for name in (
        'random_seed', 'batch_size', 'learning_rate', 'train_epochs', 'patience', 'lradj', 'pct_start')}
    digest = hashlib.sha256(json.dumps(dict(model=config, budget=budget, loss='MSE/no-SAM'),
                                      sort_keys=True).encode()).hexdigest()[:10]
    return setting + '_fact_' + digest


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fact_provenance(args):
    root = Path(__file__).resolve().parents[2]
    files = ('models/FACT.py', 'utils/fact_original/support.py', 'exp/exp_fact.py',
             'run.py', 'exp/exp_main.py', 'utils/tools.py', 'data_provider/data_factory.py',
             'data_provider/data_loader.py', 'data_provider/data_loader_gjemnessund.py',
             'utils/timefeatures.py', 'utils/gje_timefeatures.py',
             'utils/fact_original/script_sources.json')
    return dict(effective_model_config=vars(fact_model_config(args)),
                source_sha256={name: sha256_file(root / name) for name in files},
                source_manifest=json.loads((root / 'utils/fact_original/sources.json').read_text()))
