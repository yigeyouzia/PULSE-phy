"""Offline ETTh1 checkpoint diagnostics; no optimizer or checkpoint writes.

From the project root:
python -B -m notes.tools.diagnose_etth1_checkpoints

Uses the existing models and Dataset. The fixed gate is selected on validation
only. BN population calibration reads training inputs only and changes an
in-memory copy, never the saved checkpoint. This is not a training entrypoint.
"""

import argparse
import ast
from collections import defaultdict
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_hour
from models import PULSE, PULSE_FACT


ROOT = Path(__file__).resolve().parents[2]
GATES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
CHANNELS = ('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_start_state(checkpoint_root):
    """Capture this run locally; do not depend on a previous /tmp file."""
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    paths = {ROOT/name for name in names if name and (ROOT/name).is_file()}
    paths.update(checkpoint_root.glob('*/checkpoint.pth'))
    paths.update((ROOT/'all_datasets/ETTh1.csv', ROOT.parent/'AGENTS.md'))
    return dict(date='2026-11-10', head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT).decode(),
                protected_sha256={str(path):sha256(path) for path in sorted(paths)})


def checkpoint_parameter_summary(state):
    return dict(revin_gamma=dict(zip(CHANNELS, state['revin.weight'].flatten().tolist())),
                revin_beta=dict(zip(CHANNELS, state['revin.bias'].flatten().tolist())),
                codebook_rms=state['codebook'].double().square().mean().sqrt().item(),
                bn_tracked_batches={k:int(v) for k,v in state.items() if k.endswith('num_batches_tracked')})


def parse_log(path):
    text = path.read_text()
    call = ast.parse(next(s for s in text.splitlines() if s.startswith('Namespace(')), mode='eval').body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'Namespace':
        raise ValueError('Expected literal Namespace in ' + str(path))
    args = {v.arg: ast.literal_eval(v.value) for v in call.keywords}
    metrics = re.findall(r'mse:([\d.eE+-]+), mae:([\d.eE+-]+)', text)
    if len(metrics) != 1 or 'Traceback (most recent call last)' in text:
        raise ValueError('Incomplete or ambiguous experiment: ' + str(path))
    epochs = re.findall(r'Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([\d.eE+-]+)\s*Vali Loss:\s*([\d.eE+-]+)\s*Test Loss:\s*([\d.eE+-]+)', text)
    best = min(epochs, key=lambda e: float(e[3]))
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path), args=args,
                mse=float(metrics[0][0]), mae=float(metrics[0][1]),
                best_epoch=int(best[0]), steps=int(best[1]), best_logged_val=float(best[3]),
                epochs=len(epochs), logged_setting=re.search(r'>>>>>>>start training : (.*?)>+', text).group(1))


class Metrics:
    def __init__(self, horizon):
        self.horizon = horizon
        self.count = 0
        self.squared = torch.zeros(7, dtype=torch.float64)
        self.absolute = self.squared.clone()
        self.signed = self.squared.clone()
        self.window_level_squared = self.squared.clone()
        self.fluctuation_squared = self.squared.clone()
        self.lead_squared = torch.zeros(horizon, dtype=torch.float64)
        self.lead_absolute = self.lead_squared.clone()
        self.months = defaultdict(lambda: [0.0, 0.0, 0])

    def add(self, pred, target, months):
        error = pred.double() - target.double()
        if not torch.isfinite(error).all():
            raise ValueError('Non-finite prediction')
        self.count += error.shape[0] * self.horizon
        self.squared += error.square().sum((0, 1))
        self.absolute += error.abs().sum((0, 1))
        self.signed += error.sum((0, 1))
        window_mean = error.mean(dim=1, keepdim=True)
        self.window_level_squared += window_mean.square().sum((0, 1)) * self.horizon
        self.fluctuation_squared += (error-window_mean).square().sum((0, 1))
        self.lead_squared += error.square().sum((0, 2))
        self.lead_absolute += error.abs().sum((0, 2))
        for month in set(months):
            part = error[torch.tensor([v == month for v in months])]
            row = self.months[month]
            row[0] += part.square().sum().item()
            row[1] += part.abs().sum().item()
            row[2] += part.numel()

    def result(self):
        lead_count = self.count // self.horizon * 7
        boundaries = np.linspace(0, self.horizon, 5, dtype=int)
        torch.testing.assert_close(self.window_level_squared+self.fluctuation_squared,
                                   self.squared, rtol=1e-12, atol=1e-9)
        return dict(windows=self.count // self.horizon,
                    mse=(self.squared.sum() / (self.count * 7)).item(),
                    mae=(self.absolute.sum() / (self.count * 7)).item(),
                    channels={name: dict(mse=(self.squared[i] / self.count).item(),
                        mae=(self.absolute[i] / self.count).item(), bias=(self.signed[i] / self.count).item())
                        for i, name in enumerate(CHANNELS)},
                    error_decomposition=dict(
                        window_level_mse=(self.window_level_squared.sum()/(self.count*7)).item(),
                        fluctuation_mse=(self.fluctuation_squared.sum()/(self.count*7)).item(),
                        by_channel={name:dict(window_level_mse=(self.window_level_squared[i]/self.count).item(),
                            fluctuation_mse=(self.fluctuation_squared[i]/self.count).item())
                            for i,name in enumerate(CHANNELS)}),
                    lead_mse=(self.lead_squared / lead_count).tolist(),
                    lead_mae=(self.lead_absolute / lead_count).tolist(),
                    lead_quarters=[dict(start=int(a + 1), end=int(b),
                        mse=(self.lead_squared[a:b].sum() / (lead_count * (b-a))).item(),
                        mae=(self.lead_absolute[a:b].sum() / (lead_count * (b-a))).item())
                        for a, b in zip(boundaries[:-1], boundaries[1:])],
                    by_forecast_start_month={k: dict(mse=v[0]/v[2], mae=v[1]/v[2], elements=v[2])
                        for k, v in sorted(self.months.items())})


class Moments:
    def __init__(self):
        self.count = 0
        self.total = None
        self.squared = None

    def add(self, x):
        x = x.detach().double()
        total, squared = x.sum((0, 2, 3)), x.square().sum((0, 2, 3))
        self.count += x.shape[0] * x.shape[2] * x.shape[3]
        self.total = total if self.total is None else self.total + total
        self.squared = squared if self.squared is None else self.squared + squared

    def values(self):
        mean = self.total / self.count
        return mean, (self.squared / self.count - mean.square()).clamp_min(0)


def quantiles(x):
    values = x.detach().double().flatten()
    return dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                    torch.quantile(values, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.double)).tolist()))


class Components:
    def __init__(self, model):
        self.model = model
        self.bn = {name: Moments() for name, mod in model.named_modules() if isinstance(mod, nn.BatchNorm2d)}
        self.modules = dict(model.named_modules())
        self.hooks = [self.modules[name].register_forward_pre_hook(self._bn_hook(name)) for name in self.bn]
        self.updates = {}
        self.current = {}
        for index, branch in enumerate(model.fact_block.block.conv):
            self.hooks.append(branch.register_forward_hook(self._update_hook('dilation_' + str(index + 1))))
        self.hooks.append(model.fact_block.block.conv_ffn.register_forward_hook(self._update_hook('ffn')))
        self.base_squared = torch.zeros(7, dtype=torch.double)
        self.cosine_sum = 0.
        self.cosine_count = 0

    def _bn_hook(self, name):
        def hook(module, inputs):
            self.bn[name].add(inputs[0])
        return hook

    def _update_hook(self, name):
        def hook(module, inputs, output):
            self.current[name] = output.flatten(2)[:, :, :7].transpose(1, 2)
        return hook

    def add(self, hidden, interacted):
        self.base_squared += hidden.double().square().sum((0, 2))
        self.current['whole_fact_delta'] = interacted - hidden
        for name, update in self.current.items():
            value = update.double().square().sum((0, 2))
            self.updates[name] = self.updates.get(name, torch.zeros_like(value)) + value
        self.cosine_sum += torch.nn.functional.cosine_similarity(hidden, interacted, dim=-1).double().sum().item()
        self.cosine_count += hidden.shape[0] * hidden.shape[1]
        self.current.clear()

    def result(self):
        output = dict(hidden_interacted_cosine_mean=self.cosine_sum/self.cosine_count,
                      rms_update_over_hidden={name: dict(global_ratio=(value.sum()/self.base_squared.sum()).sqrt().item(),
                        by_variable=dict(zip(CHANNELS, (value/self.base_squared.clamp_min(1e-30)).sqrt().tolist())))
                        for name, value in self.updates.items()}, bn={})
        for name, moments in self.bn.items():
            module = self.modules[name]
            mean, var = moments.values()
            shift = (mean-module.running_mean.double())/(module.running_var.double()+module.eps).sqrt()
            ratio = ((var+module.eps)/(module.running_var.double()+module.eps)).sqrt()
            output['bn'][name] = dict(observations_per_feature=moments.count,
                tracked_training_batches=int(module.num_batches_tracked),
                standardized_abs_mean_shift=quantiles(shift.abs()), std_ratio=quantiles(ratio),
                stored_gamma=quantiles(module.weight), stored_beta=quantiles(module.bias))
        return output

    def close(self):
        for hook in self.hooks:
            hook.remove()


def dataset(args, flag):
    return Dataset_ETT_hour(root_path=args.root_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
        data_path=args.data_path, target=args.target, timeenc=1 if args.embed == 'timeF' else 0,
        freq=args.freq, time_dim=args.time_dim)


def loader(data, batch_size):
    return DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)


def hidden_parts(model, batch):
    x, _, xm, _, cycle = batch
    timestamp_x = model.time_enc_x(xm.float())
    x_in, anchor, mean, var = model.norm(x.float(), cycle.int(), timestamp_x)
    hidden = model.backbone[1](model.backbone[0](x_in.transpose(1, 2)))
    return hidden, anchor, mean, var


def decode(model, hidden, anchor, mean, var, timestamp_y):
    y = model.backbone[2](hidden).transpose(1, 2)
    return model.denorm(y, anchor, mean, var, timestamp_y)


def evaluate(model, args, flag, batch_size, raw_dates, gates=None, components=False):
    data = dataset(args, flag)
    values = {alpha: Metrics(args.pred_len) for alpha in (gates or (1.0,))}
    observer = Components(model) if components else None
    first_target = 8640 if flag == 'val' else 11520
    offset = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in loader(data, batch_size):
                hidden, anchor, mean, var = hidden_parts(model, batch)
                interacted = model.fact_block(hidden) if hasattr(model, 'fact_block') else hidden
                if observer:
                    observer.add(hidden, interacted)
                timestamp_y = model.time_enc_y(batch[3].float())
                count = hidden.shape[0]
                months = raw_dates.iloc[first_target+offset:first_target+offset+count].dt.strftime('%Y-%m').tolist()
                target = batch[1][:, -args.pred_len:, :].float()
                for alpha, metric in values.items():
                    mixed = interacted if alpha == 1 else hidden + alpha*(interacted-hidden)
                    metric.add(decode(model, mixed, anchor, mean, var, timestamp_y), target, months)
                offset += count
        result = dict(metrics={str(k): v.result() for k, v in values.items()},
                      seconds=time.perf_counter()-started, samples=offset)
        if observer:
            result['components'] = observer.result()
    finally:
        if observer:
            observer.close()
    print('Evaluated', args.model, args.pred_len, flag, offset, 'windows;', round(result['seconds'], 2), 's', flush=True)
    return result


def calibrate_bn(model, args, batch_size):
    """Two stages of exact training-input population moments, no target use."""
    modules = dict(model.named_modules())
    branches = [name for name, mod in modules.items() if isinstance(mod, nn.BatchNorm2d) and '.conv.' in name]
    ffn = [name for name, mod in modules.items() if isinstance(mod, nn.BatchNorm2d) and '.conv_ffn.' in name]
    changes = {}
    for stage in (branches, ffn):
        moments = {name: Moments() for name in stage}
        hooks = []
        for name in stage:
            def hook(module, inputs, name=name):
                moments[name].add(inputs[0])
            hooks.append(modules[name].register_forward_pre_hook(hook))
        try:
            with torch.inference_mode():
                for batch in loader(dataset(args, 'train'), batch_size):
                    hidden, _, _, _ = hidden_parts(model, batch)
                    model.fact_block(hidden)
        finally:
            for handle in hooks:
                handle.remove()
        with torch.no_grad():
            for name, moment in moments.items():
                module = modules[name]
                mean, var = moment.values()
                changes[name] = dict(observations=moment.count,
                    normalized_mean_change=quantiles(((mean-module.running_mean.double())/(module.running_var.double()+module.eps).sqrt()).abs()),
                    std_ratio=quantiles(((var+module.eps)/(module.running_var.double()+module.eps)).sqrt()))
                module.running_mean.copy_(mean.float())
                module.running_var.copy_(var.float())
    print('BN population calibration complete', args.pred_len, flush=True)
    return changes


def verify_parts(model, args):
    batch = next(iter(loader(dataset(args, 'test'), 2)))
    with torch.inference_mode():
        native = model(batch[0].float(), batch[4].int(), batch[2].float(), batch[3].float(), do_mixup=False)
        hidden, anchor, mean, var = hidden_parts(model, batch)
        if hasattr(model, 'fact_block'):
            hidden = model.fact_block(hidden)
        decoded = decode(model, hidden, anchor, mean, var, model.time_enc_y(batch[3].float()))
        torch.testing.assert_close(decoded, native, rtol=0, atol=0)


def source_audit(start):
    return [path for path, digest in start['protected_sha256'].items()
            if not Path(path).is_file() or sha256(Path(path)) != digest]


def write_outputs(report, output_dir):
    (output_dir/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (output_dir/'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'horizon', 'variant', 'split', 'alpha', 'mse', 'mae', 'windows'])
        for row in report['runs']:
            for variant in ('original', 'bn_recalibrated'):
                for split, data in row.get(variant, {}).items():
                    for alpha, metric in data['metrics'].items():
                        writer.writerow([row['model'], row['horizon'], variant, split, alpha,
                                         metric['mse'], metric['mae'], metric['windows']])
    with (output_dir/'channel_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'horizon', 'variant', 'split', 'alpha', 'channel', 'mse', 'mae', 'bias'])
        for row in report['runs']:
            for variant in ('original', 'bn_recalibrated'):
                for split, data in row.get(variant, {}).items():
                    for alpha, metric in data['metrics'].items():
                        for name, value in metric['channels'].items():
                            writer.writerow([row['model'],row['horizon'],variant,split,alpha,name,
                                             value['mse'], value['mae'], value['bias']])
    with (output_dir/'error_decomposition.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','horizon','variant','split','alpha','channel','window_level_mse','fluctuation_mse'])
        for row in report['runs']:
            for variant in ('original','bn_recalibrated'):
                for split,data in row.get(variant,{}).items():
                    for alpha,metric in data['metrics'].items():
                        value=metric.get('error_decomposition')
                        if value is None:
                            continue
                        writer.writerow([row['model'],row['horizon'],variant,split,alpha,'ALL',
                                         value['window_level_mse'],value['fluctuation_mse']])
                        for name,part in value['by_channel'].items():
                            writer.writerow([row['model'],row['horizon'],variant,split,alpha,name,
                                             part['window_level_mse'],part['fluctuation_mse']])


def main(options):
    torch.set_num_threads(options.threads)
    torch.manual_seed(2024)
    np.random.seed(2024)
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = json.loads(options.start_snapshot.read_text()) if options.start_snapshot else capture_start_state(options.checkpoint_root)
    (output_dir/'start_state.json').write_text(json.dumps(start, ensure_ascii=False, indent=2)+'\n')
    raw_path = ROOT/'all_datasets/ETTh1.csv'
    raw = pd.read_csv(raw_path, parse_dates=['date'])
    report = dict(date='2026-11-10', purpose='Frozen-checkpoint diagnosis, not new model training',
        environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                         numpy=np.__version__, pandas=pd.__version__, device='cpu', threads=options.threads),
        local_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        checkpoint_root=str(options.checkpoint_root), data_source=dict(path=str(raw_path),sha256=sha256(raw_path)),
        gate_candidates=list(GATES), gate_selection='minimum full validation MSE only; fixed weights',
        validation_policy='ordered complete val; differs from original shuffled drop_last validation',
        bn_policy='training-input population moments only, two stages; no SAM or dropout; no parameter training',
        limitations=['one historical seed per task; not a causal architecture ablation',
                     'no PC or STATIC checkpoints in this batch',
                     'server source revision/environment/data digest were not logged',
                     'posthoc alpha=0 does not equal separately trained PULSE'], runs=[])
    train = raw.loc[:8639, list(CHANNELS)].to_numpy()
    report['raw_segment_shift'] = {name: dict(zip(CHANNELS, ((raw.loc[a:b-1,list(CHANNELS)].to_numpy().mean(0)-train.mean(0))/train.std(0)).tolist()))
                                   for name,a,b in [('val',8640,11520),('test',11520,14400)]}
    models_by_horizon = {}
    for horizon in (96, 720):
        for name, module, log_folder in (('PULSE', PULSE, 'Pulse'), ('PULSE_FACT', PULSE_FACT, 'PULSE_FACT')):
            log_name = 'ETTh1' if name == 'PULSE' else name
            log = parse_log(ROOT/'logs'/log_folder/'ETTh1'/f'{log_name}_96_{horizon}.log')
            args = SimpleNamespace(**log['args'])
            if args.data != 'ETTh1' or args.features != 'M' or args.seq_len != 96 or args.pred_len != horizon:
                raise ValueError('Unexpected task configuration')
            paths = list(options.checkpoint_root.glob(f'{name}_ETTh1_96_{horizon}_*/checkpoint.pth'))
            if len(paths) != 1:
                raise ValueError('Expected one matching checkpoint: ' + str(paths))
            path = paths[0]
            with contextlib.redirect_stdout(io.StringIO()):
                model = module.Model(args).eval()
            state = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(state, strict=True)
            verify_parts(model, args)
            row = dict(model=name,horizon=horizon,log=log,checkpoint=dict(path=str(path.relative_to(ROOT)),
                       sha256=sha256(path),strict_load=True,extra_state=state.get('_extra_state'),
                       trainable_parameters=sum(v.numel() for v in model.parameters()),
                       parameter_summary=checkpoint_parameter_summary(state)),
                       decomposed_forward_exact=True,original={})
            gates = GATES if name == 'PULSE_FACT' else None
            row['original']['val'] = evaluate(model,args,'val',options.batch_size,raw.date,gates,components=name=='PULSE_FACT')
            best = min(row['original']['val']['metrics'],key=lambda k:row['original']['val']['metrics'][k]['mse'])
            row['validation_selected_alpha'] = float(best)
            test_gates = tuple(sorted(set((1.,float(best))))) if name=='PULSE_FACT' else None
            row['original']['test'] = evaluate(model,args,'test',options.batch_size,raw.date,test_gates,components=name=='PULSE_FACT')
            metric = row['original']['test']['metrics']['1.0']
            row['log_replay'] = {k+'_absolute_difference':abs(metric[k]-log[k]) for k in ('mse','mae')}
            row['log_replay']['within_1e_5'] = all(v<1e-5 for v in row['log_replay'].values())
            if not row['log_replay']['within_1e_5']:
                raise ValueError('Checkpoint/log/data replay mismatch: '+str(row['log_replay']))
            models_by_horizon[horizon,name] = {k:v.detach().clone() for k,v in model.state_dict().items() if torch.is_tensor(v)}
            if name == 'PULSE_FACT':
                calibrated = copy.deepcopy(model).eval()
                row['bn_calibration_changes'] = calibrate_bn(calibrated,args,options.batch_size)
                row['bn_recalibrated'] = {split:evaluate(calibrated,args,split,options.batch_size,raw.date)
                                           for split in ('val','test')}
                base = models_by_horizon[horizon,'PULSE']
                grouped = defaultdict(lambda:[0.,0.])
                for key,value in model.state_dict().items():
                    if key in base:
                        group=key.split('.')[0]
                        grouped[group][0]+=(value.double()-base[key].double()).square().sum().item()
                        grouped[group][1]+=base[key].double().square().sum().item()
                row['shared_parameter_relative_l2_difference']={k:(v[0]/max(v[1],1e-30))**.5 for k,v in grouped.items()}
            report['runs'].append(row)
            write_outputs(report,output_dir)
            print('COMPLETED',name,horizon,'test',metric['mse'],metric['mae'],'selected alpha',best,flush=True)
    report['protected_file_changes'] = source_audit(start)
    if report['protected_file_changes']:
        raise ValueError('Protected files changed: '+str(report['protected_file_changes']))
    report['analysis_script_sha256'] = sha256(Path(__file__).resolve())
    report['all_checks_complete'] = True
    write_outputs(report,output_dir)
    print('All four checkpoint replays and protected-file checks passed.',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root',type=Path,default=ROOT/'checkpoints/pulse+pulsefact')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'notes/data/etth1_checkpoint_diagnostics_20261110')
    parser.add_argument('--start-snapshot',type=Path,default=None,
                        help='Optional existing protection snapshot; otherwise capture this workspace at start')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    options = parser.parse_args()
    if options.batch_size < 1 or options.threads < 1:
        parser.error('batch-size and threads must be positive')
    main(options)
