"""Complete FACT under the existing data/budget protocol and native MSE loss.

The model is copied byte-for-byte. This experiment adapts invocation, training
targets and reporting; legacy experiment implementations are not modified.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR

from exp.exp_main import Exp_Main
from models.FACT import Model
from utils.fact_original.support import (fact_model_config, fact_provenance,
                                         sha256_file, validate_fact_arguments)
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_FACT(Exp_Main):
    def _build_model(self):
        validate_fact_arguments(self.args)
        self.fact_config = fact_model_config(self.args)
        print('FACT effective configuration: ' + json.dumps(vars(self.fact_config), sort_keys=True), flush=True)
        print('FACT protocol: existing data/budget; MSE; no SAM; no FFT auxiliary loss', flush=True)
        return Model(self.fact_config).float()

    def _forward_fact(self, x):
        # freq=n preserves the source's real-variable-only path. No target,
        # future signal, cycle or replacement PULSE module enters the model.
        return self.model(x, None, None, None)

    def _fact_data_signature(self, data):
        if self.args.data == 'Gjemnessund':
            return data.info
        source = Path(self.args.root_path) / self.args.data_path
        scaler = data.scaler
        return dict(dataset=self.args.data, source_sha256=sha256_file(source),
                    seq_len=self.args.seq_len, pred_len=self.args.pred_len,
                    features=self.args.features, test_windows=len(data),
                    scaler_mean=np.asarray(scaler.mean_).tolist(),
                    scaler_scale=np.asarray(scaler.scale_).tolist())

    def _save_fact_protocol(self, path, datasets):
        path = Path(path)
        train, val, test = datasets
        if self.args.data == 'Gjemnessund':
            for data in datasets:
                data.export_protocol(path)
        signature = self._fact_data_signature(test)
        (path / 'fact_data_protocol.json').write_text(json.dumps(signature, indent=2), encoding='utf-8')
        info = dict(args=vars(self.args), **fact_provenance(self.args),
                    python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                    device=str(self.device), train_windows=len(train), val_windows=len(val),
                    test_windows=len(test), parameter_count=sum(p.numel() for p in self.model.parameters()),
                    loss='MSE; rec_lambda=1; auxi_lambda=0; no SAM',
                    scheduler='Existing OneCycleLR construction and utils.tools lradj behavior',
                    test_policy='Validation selects checkpoint; test only after training',
                    debug=bool(self.args.data == 'Gjemnessund' and
                               (self.args.gje_record_limit or self.args.gje_max_batches)))
        (path / 'fact_run.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        test_data, _ = self._get_data('test')
        if not len(train_loader) or not len(val_loader):
            raise ValueError('FACT train/validation loader is empty; check lengths and batch_size')
        path = Path(self.args.checkpoints) / setting
        path.mkdir(parents=True, exist_ok=True)
        self._save_fact_protocol(path, (train_data, val_data, test_data))
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = OneCycleLR(optimizer, steps_per_epoch=len(train_loader),
                               pct_start=self.args.pct_start, epochs=self.args.train_epochs,
                               max_lr=self.args.learning_rate)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        interval = self.args.gje_log_interval if self.args.data == 'Gjemnessund' else 100
        for epoch in range(self.args.train_epochs):
            self.model.train()
            losses = []
            started = time.time()
            print('Effective epoch learning rate:', optimizer.param_groups[0]['lr'], flush=True)
            for i, (x, y, xm, ym, cycle) in enumerate(train_loader):
                if self._gje_batch_limit(i):
                    break
                x = x.float().to(self.device)
                target = y[:, -self.args.pred_len:, :].float().to(self.device)
                optimizer.zero_grad()
                prediction = self._forward_fact(x)
                loss = criterion(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite FACT training loss')
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if (i + 1) % interval == 0:
                    print('\titers: {}, epoch: {} | loss_rec: {:.7f}'.format(i + 1, epoch + 1, loss.item()), flush=True)
                if self.args.lradj == 'TST':
                    adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            if not losses:
                raise ValueError('FACT training did not evaluate any batches')
            # Reuse legacy validation, including Gje's weighted tail batches.
            val_loss = self.vali(val_data, val_loader, criterion)
            if not np.isfinite(val_loss):
                raise ValueError('Non-finite FACT validation loss')
            print('Epoch: {} cost time: {:.3f}'.format(epoch + 1, time.time() - started), flush=True)
            print('Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}'.format(
                epoch + 1, len(losses), float(np.mean(losses)), val_loss), flush=True)
            early_stopping(val_loss, self.model, str(path))
            if early_stopping.early_stop:
                print('Early stopping', flush=True)
                break
            if self.args.lradj != 'TST':
                adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args)
        self.model.load_state_dict(torch.load(str(path / 'checkpoint.pth'), map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        data, loader = self._get_data('test')
        path = Path(self.args.checkpoints) / setting
        saved = json.loads((path / 'fact_run.json').read_text(encoding='utf-8'))
        if saved['effective_model_config'] != vars(self.fact_config):
            raise ValueError('FACT checkpoint model configuration differs from this run')
        protocol = json.loads((path / 'fact_data_protocol.json').read_text(encoding='utf-8'))
        if protocol != self._fact_data_signature(data):
            raise ValueError('FACT test data/scaler differs from saved training protocol')
        if test:
            self.model.load_state_dict(torch.load(str(path / 'checkpoint.pth'), map_location=self.device))
        self.model.eval()
        channels = self.args.enc_in
        squared = np.zeros(channels, dtype=np.float64)
        absolute = np.zeros(channels, dtype=np.float64)
        count, cursor = 0, 0
        is_gje = self.args.data == 'Gjemnessund'
        if is_gje:
            record_squared = np.zeros((len(data.records), channels), dtype=np.float64)
            record_absolute = np.zeros_like(record_squared)
            record_count = np.zeros(len(data.records), dtype=np.int64)
        started = time.time()
        with torch.no_grad():
            for i, (x, y, xm, ym, cycle) in enumerate(loader):
                if self._gje_batch_limit(i):
                    break
                prediction = self._forward_fact(x.float().to(self.device))
                target = y[:, -self.args.pred_len:, :].float()
                error = prediction.cpu().numpy().astype(np.float64) - target.numpy().astype(np.float64)
                if error.shape[1:] != (self.args.pred_len, channels) or not np.isfinite(error).all():
                    raise ValueError('Invalid FACT test shape or non-finite predictions')
                batch_sq = np.square(error).sum(axis=1)
                batch_abs = np.abs(error).sum(axis=1)
                squared += batch_sq.sum(axis=0)
                absolute += batch_abs.sum(axis=0)
                if is_gje:
                    ids = data.windows[cursor:cursor + len(error), 0]
                    np.add.at(record_squared, ids, batch_sq)
                    np.add.at(record_absolute, ids, batch_abs)
                    np.add.at(record_count, ids, error.shape[1])
                count += len(error) * error.shape[1]
                cursor += len(error)
        if not count:
            raise ValueError('FACT test loader evaluated no windows')
        per_mse, per_mae = squared / count, absolute / count
        report = dict(model='FACT', mse=float(per_mse.mean()), mae=float(per_mae.mean()),
                      evaluated_windows=cursor, elapsed_seconds=time.time() - started,
                      effective_model_config=vars(self.fact_config),
                      debug=bool(is_gje and (self.args.gje_record_limit or self.args.gje_max_batches)),
                      metric_scale='shared train StandardScaler coordinates, after FACT internal denorm',
                      protocol='Common data/budget; MSE; no SAM; not paper metric reproduction')
        out = Path('results') / setting
        out.mkdir(parents=True, exist_ok=True)
        if is_gje:
            used = record_count > 0
            macro_mse = (record_squared[used] / record_count[used, None]).mean(axis=0)
            macro_mae = (record_absolute[used] / record_count[used, None]).mean(axis=0)
            report.update(acceleration_mse=float(per_mse[:6].mean()),
                          acceleration_mae=float(per_mae[:6].mean()),
                          wind_mse=float(per_mse[6:].mean()), wind_mae=float(per_mae[6:].mean()),
                          macro_record_mse=float(macro_mse.mean()), macro_record_mae=float(macro_mae.mean()),
                          evaluated_records=int(used.sum()),
                          per_channel={name: dict(mse=float(per_mse[j]), mae=float(per_mae[j]))
                                       for j, name in enumerate(data.columns)},
                          angle_note='Raw wind angle; Euclidean error has circular-boundary limitations')
            data.export_protocol(out)
            (out / 'gje_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        else:
            report['per_channel'] = [dict(mse=float(m), mae=float(a)) for m, a in zip(per_mse, per_mae)]
        (out / 'fact_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('mse:{}, mae:{}'.format(report['mse'], report['mae']), flush=True)
        if is_gje:
            print('Acc6 mse:{}, mae:{}'.format(report['acceleration_mse'], report['acceleration_mae']), flush=True)
        with open('result.txt', 'a') as stream:
            stream.write(setting + '\n' + json.dumps(report) + '\n\n')
        return report
