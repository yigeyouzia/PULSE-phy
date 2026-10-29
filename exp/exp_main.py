from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from models import Autoformer, DLinear, PatchTST,iTransformer,CycleNet,TimeXer,TimesNet,PULSE
from models import PULSE_SOFTS, PULSE_SCF_SIMPLE, PULSE_SCF
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch)
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import json
import hashlib
import sys
from pathlib import Path

import warnings
import numpy as np

warnings.filterwarnings('ignore')



class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # self.kmeans_tool = KMeansInitializer(K=7, seq_len=168, device=self.device)
        if args.add_noise and args.noise_amp > 0:
            seq_len = args.pred_len
            cutoff_freq_percentage = args.noise_freq_percentage
            cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
            if args.auxi_mode == "rfft":
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1).to(self.device)
        else:
            self.mask = None


    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'PULSE': PULSE,
            'PULSE_SOFTS': PULSE_SOFTS,
            'PULSE_SCF_SIMPLE': PULSE_SCF_SIMPLE,
            'PULSE_SCF': PULSE_SCF,
            'iTransformer':iTransformer,
            'CycleNet':CycleNet,
            'TimeXer':TimeXer,
            'TimesNet':TimesNet,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        weighted_sum, weighted_count = 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,cycle_index) in enumerate(vali_loader):
                if self._gje_batch_limit(i):
                    break
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                cycle_index = cycle_index.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'PULSE'}):
                            outputs = self.model(batch_x, cycle_index,batch_x_mark=batch_x_mark,batch_y_mark=batch_y_mark)

                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'PULSE'}):
                        outputs = self.model(batch_x, cycle_index,batch_x_mark=batch_x_mark,batch_y_mark=batch_y_mark)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST',}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                criterion= nn.MSELoss()
                loss = criterion(pred, true)

                total_loss.append(loss)
                if getattr(vali_data, 'weighted_evaluation', False):
                    weighted_sum += loss.item() * pred.numel()
                    weighted_count += pred.numel()
        total_loss = weighted_sum / weighted_count if weighted_count else np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.data == 'Gjemnessund':
            self._save_gje_protocol(path, (train_data, vali_data, test_data))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            if self.args.data == 'Gjemnessund':
                print('Effective epoch learning rate:', model_optim.param_groups[0]['lr'], flush=True)
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index) in enumerate(train_loader):
                if self._gje_batch_limit(i):
                    break
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                cycle_index = cycle_index.int().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'PULSE'}):
                            outputs,inv= self.model(batch_x, cycle_index,batch_x_mark=batch_x_mark,batch_y_mark=batch_y_mark)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'PULSE'}):
                        outputs,mixup_params= self.model(batch_x, cycle_index,batch_x_mark=batch_x_mark,batch_y_mark=batch_y_mark)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST', }):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if mixup_params is not None:
                        lam, index = mixup_params

                        batch_y= lam * batch_y + (1 - lam) * batch_y[index, :]
                    loss = 0


                    loss += 0
                    if self.args.rec_lambda:
                        loss_rec = criterion(outputs, batch_y)

                        loss += self.args.rec_lambda * loss_rec
                        if (i + 1) % (self.args.gje_log_interval if self.args.data == 'Gjemnessund' else 100) == 0:
                            print(f"\tloss_rec: {loss_rec.item()}")


                    if self.args.auxi_lambda:
                        # fft shape: [B, P, D]
                        if self.args.auxi_mode == "fft":
                            loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)

                        elif self.args.auxi_mode == "rfft":
                            if self.args.auxi_type == 'complex':
                                loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
                        if self.args.auxi_loss == "MAE":
                            loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                        elif self.args.auxi_loss == "MSE":
                            loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                        else:
                            raise NotImplementedError

                        loss += self.args.auxi_lambda * loss_auxi
                        if (i + 1) % (self.args.gje_log_interval if self.args.data == 'Gjemnessund' else 100) == 0:
                            print(f"\tloss_auxi: {loss_auxi.item()}")


                    train_loss.append(loss.item())

                if (i + 1) % (self.args.gje_log_interval if self.args.data == 'Gjemnessund' else 100) == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()


                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.data == 'Gjemnessund':
                print("Epoch: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    epoch + 1, train_loss, vali_loss), flush=True)
            else:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.data == 'Gjemnessund':
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage.cuda(self.args.gpu)))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if self.args.data == 'Gjemnessund':
            return self._test_gjemnessund(setting, test_data, test_loader, test)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=lambda storage, loc: storage.cuda(0)))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index,) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                cycle_index = cycle_index.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'PULSE'}):
                            outputs = self.model(batch_x, cycle_index)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'PULSE'}):
                        outputs = self.model(batch_x, cycle_index,batch_x_mark=batch_x_mark,batch_y_mark=batch_y_mark)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                #     np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                #     np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                cycle_index = cycle_index.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'PULSE'}):
                            outputs = self.model(batch_x, cycle_index,)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'PULSE'}):
                        outputs = self.model(batch_x, cycle_index,batch_x_mark=batch_x_mark,batch_y_mark=batch_y_mark)

                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    def _gje_batch_limit(self, index):
        return (self.args.data == 'Gjemnessund' and self.args.gje_max_batches > 0
                and index >= self.args.gje_max_batches)

    def _save_gje_protocol(self, path, datasets):
        for dataset in datasets:
            dataset.export_protocol(path)
        info = dict(args=vars(self.args), python=sys.version, torch=torch.__version__,
                    numpy=np.__version__, device=str(self.device),
                    model_sha256=hashlib.sha256(Path('models/PULSE.py').read_bytes()).hexdigest(),
                    debug=bool(self.args.gje_record_limit or self.args.gje_max_batches),
                    loss='Original non-AMP SAM target mixing + configured rec/FFT loss',
                    scheduler='Original OneCycleLR construction + original lradj behavior',
                    test_policy='No test-set evaluation/selection during training')
        (Path(path) / 'gje_run.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

    def _test_gjemnessund(self, setting, data, loader, load):
        path = Path(self.args.checkpoints) / setting
        protocol_file = path / 'gje_data_protocol.json'
        if not protocol_file.is_file():
            raise FileNotFoundError('Missing training protocol: ' + str(protocol_file))
        saved = json.loads(protocol_file.read_text(encoding='utf-8'))
        if saved != data.info:
            raise ValueError('Gje test data/scaler differs from the saved training protocol.')
        if load:
            self.model.load_state_dict(torch.load(str(path / 'checkpoint.pth'), map_location=self.device))
        self.model.eval()
        squared = np.zeros(10, dtype=np.float64)
        absolute = np.zeros(10, dtype=np.float64)
        record_squared = np.zeros((len(data.records), 10), dtype=np.float64)
        record_absolute = np.zeros_like(record_squared)
        record_count = np.zeros(len(data.records), dtype=np.int64)
        count, cursor = 0, 0
        started = time.time()
        with torch.no_grad():
            for i, (x, y, xm, ym, cycle) in enumerate(loader):
                if self._gje_batch_limit(i):
                    break
                x, xm, ym = (a.float().to(self.device) for a in (x, xm, ym))
                pred = self.model(x, cycle.int().to(self.device), xm, ym)
                error = pred.cpu().numpy().astype(np.float64) - y.numpy().astype(np.float64)
                if not np.isfinite(error).all():
                    raise ValueError('Non-finite Gje predictions; no metrics saved.')
                batch_sq = np.square(error).sum(axis=1)
                batch_abs = np.abs(error).sum(axis=1)
                squared += batch_sq.sum(axis=0)
                absolute += batch_abs.sum(axis=0)
                ids = data.windows[cursor:cursor + len(error), 0]
                np.add.at(record_squared, ids, batch_sq)
                np.add.at(record_absolute, ids, batch_abs)
                np.add.at(record_count, ids, error.shape[1])
                cursor += len(error)
                count += len(error) * error.shape[1]
        used = record_count > 0
        per_mse, per_mae = squared / count, absolute / count
        macro_mse = (record_squared[used] / record_count[used, None]).mean(axis=0)
        macro_mae = (record_absolute[used] / record_count[used, None]).mean(axis=0)
        report = dict(mse=float(per_mse.mean()), mae=float(per_mae.mean()),
                      acceleration_mse=float(per_mse[:6].mean()),
                      acceleration_mae=float(per_mae[:6].mean()),
                      wind_mse=float(per_mse[6:].mean()), wind_mae=float(per_mae[6:].mean()),
                      macro_record_mse=float(macro_mse.mean()), macro_record_mae=float(macro_mae.mean()),
                      per_channel={c: dict(mse=float(per_mse[j]), mae=float(per_mae[j]))
                                   for j, c in enumerate(data.columns)},
                      evaluated_windows=cursor, evaluated_records=int(used.sum()),
                      elapsed_seconds=time.time() - started,
                      debug=bool(self.args.gje_record_limit or self.args.gje_max_batches),
                      metric_scale='shared train StandardScaler coordinates, after PULSE denorm',
                      angle_note='Raw angle retained for M10 alignment; Euclidean error has circular-boundary limitations')
        out = Path('results') / setting
        out.mkdir(parents=True, exist_ok=True)
        (out / 'gje_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        data.export_protocol(out)
        print('mse:{}, mae:{} | Acc6 mse:{}, mae:{}'.format(
            report['mse'], report['mae'], report['acceleration_mse'], report['acceleration_mae']), flush=True)
        with open('result.txt', 'a') as f:
            f.write(setting + '\n' + json.dumps(report) + '\n\n')
        return report
