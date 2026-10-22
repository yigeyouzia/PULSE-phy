"""Run: python -m unittest discover -s tests -p 'test_gjemnessund.py' -v"""
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch

from data_provider.data_loader_gjemnessund import CHANNELS, Dataset_Gjemnessund
from data_provider import data_factory
from utils.gje_timefeatures import gje_time_features


def make_h5(path, future_shift=0, irregular=False):
    rows = []
    with h5py.File(str(path), 'w') as f:
        for i, n in enumerate((120, 120, 12)):
            dates = pd.date_range('2020-02-29 10:00:00', periods=n, freq='500ms') + pd.Timedelta(minutes=5*i)
            strings = [str(x) for x in dates]
            if irregular and i == 0:
                strings[1] = str(dates[1] + pd.Timedelta(milliseconds=100))
            x = np.arange(n*10, dtype=np.float64).reshape(n,10)/10 + i*100
            x[n//2:] += future_shift
            rows.append(x)
            g = f.create_group('record_%02d' % i)
            g.create_dataset('axis0', data=np.asarray(['date']+CHANNELS,dtype='S32'))
            g.create_dataset('axis1',data=np.arange(n))
            g.create_dataset('block0_items',data=np.asarray(CHANNELS,dtype='S32'))
            g.create_dataset('block0_values',data=x)
            g.create_dataset('block1_items',data=np.asarray(['date'],dtype='S4'))
            payload = np.frombuffer(pickle.dumps(np.asarray(strings,dtype=object).reshape(n,1),protocol=2),dtype=np.uint8)
            ds = g.create_dataset('block1_values',(1,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
            ds[0] = payload
    return rows


class GjeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = make_h5(self.root/'gjemnessund.h5')

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, flag='train', **kw):
        return Dataset_Gjemnessund(str(self.root),flag=flag,size=[24,0,12],
                                  window_stride=6,cycle_len=24,**kw)

    def test_training_only_scaler_and_frozen_reuse(self):
        train = self.dataset()
        expected = np.concatenate([x[:len(x)//2] for x in self.rows])
        np.testing.assert_allclose(train.mean,expected.mean(axis=0))
        np.testing.assert_allclose(train.scale,expected.std(axis=0))
        make_h5(self.root/'future_changed.h5',future_shift=10000)
        other = Dataset_Gjemnessund(str(self.root),data_path='future_changed.h5',size=[24,0,12],window_stride=6)
        np.testing.assert_array_equal(train.mean,other.mean)
        np.testing.assert_array_equal(train.scale,other.scale)
        with patch('data_provider.data_loader_gjemnessund.StandardScaler.partial_fit',side_effect=AssertionError('Refit')):
            self.dataset('val')
            self.dataset('test')

    def test_window_boundaries_and_short_records(self):
        for flag, expected in [('train',10),('val',8),('test',8)]:
            ds=self.dataset(flag)
            self.assertEqual(len(ds),expected)
            self.assertEqual(ds.skipped,['record_02'])
            for rid,start in ds.windows:
                record=ds.records[rid]
                low,high={'train':(0,record['train_end']), 'val':(record['train_end'],record['val_end']),
                          'test':(record['val_end'],record['nrows'])}[flag]
                self.assertGreaterEqual(start,0)
                self.assertGreaterEqual(start+24,low)
                self.assertLessEqual(start+36,high)
            x,y,xm,ym,cycle=ds[0]
            self.assertEqual((x.shape,y.shape,xm.shape,ym.shape),((24,10),(12,10),(24,6),(12,6)))
            np.testing.assert_allclose(ds.inverse_transform(x),self.rows[0][ds.windows[0,1]:ds.windows[0,1]+24],atol=1e-5)
            self.assertTrue(0 <= cycle < 24)
            ds.export_protocol(self.root/'export')
            np.testing.assert_array_equal(np.load(self.root/'export'/('gje_%s_windows.npy'%flag)),ds.windows)

    def test_time_grid_and_leap_year(self):
        dates=pd.to_datetime(['2020-02-29 23:59:59.500','2020-03-01 00:00:00.000'])
        marks=gje_time_features(dates)
        self.assertEqual(marks.shape,(2,6))
        self.assertFalse(np.array_equal(marks[0],marks[1]))
        self.assertAlmostEqual(float(marks[1,4]),np.sin(2*np.pi*60/366),places=6)
        make_h5(self.root/'irregular.h5',irregular=True)
        with self.assertRaisesRegex(ValueError,'regular'):
            Dataset_Gjemnessund(str(self.root),data_path='irregular.h5',size=[24,0,12])

    def test_original_pulse_forward_backward(self):
        from models.PULSE import Model
        cfg=SimpleNamespace(seq_len=24,pred_len=12,enc_in=10,d_model=16,inv_len=24,
                            patch_size=6,time_dim=6,dsa=1,dsb=1,ksize=5)
        model=Model(cfg)
        ds=self.dataset()
        batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=2)))
        x,y,xm,ym,cycle=batch
        pred,(lam,idx)=model(x,cycle,xm,ym)
        target=lam*y+(1-lam)*y[idx]
        (torch.fft.fft(pred,dim=1)-torch.fft.fft(target,dim=1)).abs().mean().backward()
        self.assertTrue(torch.isfinite(model.codebook.grad).all())
        model.eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(x,cycle,xm,ym).shape),(2,12,10))

    def test_overlap_prefix_is_owned_once_and_recorded(self):
        path=self.root/'gjemnessund.h5'
        with h5py.File(str(path),'r+') as f:
            dates=pd.date_range('2020-02-29 10:00:59',periods=120,freq='500ms')
            strings=[str(x) for x in dates]
            payload=np.frombuffer(pickle.dumps(np.asarray(strings,dtype=object).reshape(120,1),protocol=2),dtype=np.uint8)
            f['record_01/block1_values'][0]=payload
        ds=self.dataset()
        self.assertEqual(ds.records[1]['source_start_row'],2)
        self.assertEqual(ds.records[1]['nrows'],118)
        self.assertGreater(ds.records[1]['start_ns'],ds.records[0]['end_ns'])
        expected=np.concatenate([self.rows[0][:60],self.rows[1][2:2+59],self.rows[2][:6]])
        np.testing.assert_allclose(ds.mean,expected.mean(axis=0))

    def test_nonfinite_source_is_rejected(self):
        with h5py.File(str(self.root/'gjemnessund.h5'),'r+') as f:
            f['record_00/block0_values'][5,2]=np.nan
        with self.assertRaisesRegex(ValueError,'NaN/Inf'):
            self.dataset()

    def test_legacy_factory_dispatch(self):
        # All old dataset registrations/loader options must remain unchanged.
        class Fake(torch.utils.data.Dataset):
            def __init__(self,**kwargs):self.kwargs=kwargs
            def __len__(self):return 17
            def __getitem__(self,i):return i
        args=SimpleNamespace(embed='timeF',freq='h',batch_size=4,num_workers=0,
                             root_path='old-root',data_path='old.csv',seq_len=24,label_len=0,
                             pred_len=12,features='M',target='OT',time_dim=1)
        for name in list(data_factory.data_dict):
            args.data=name
            with patch.dict(data_factory.data_dict,{name:Fake}):
                for flag in ['train','val','test']:
                    ds,loader=data_factory.data_provider(args,flag)
                    self.assertEqual(ds.kwargs['time_dim'],1)
                    self.assertEqual(ds.kwargs['freq'],'h')
                    self.assertEqual(loader.drop_last,flag!='test')
                    self.assertIsInstance(loader.sampler,torch.utils.data.SequentialSampler if flag=='test' else torch.utils.data.RandomSampler)
                    self.assertEqual(len(loader),5 if flag=='test' else 4)

    def test_validation_tail_batch_is_weighted_only_for_new_dataset(self):
        from exp.exp_main import Exp_Main
        class Zero(torch.nn.Module):
            def forward(self,x,cycle,batch_x_mark,batch_y_mark):
                return torch.zeros(x.shape[0],12,10)
        y=torch.ones(3,12,10)
        y[-1]*=3
        ds=torch.utils.data.TensorDataset(torch.zeros(3,24,10),y,
            torch.zeros(3,24,6),torch.zeros(3,12,6),torch.zeros(3,dtype=torch.int64))
        loader=torch.utils.data.DataLoader(ds,batch_size=2)
        exp=Exp_Main.__new__(Exp_Main)
        exp.args=SimpleNamespace(data='Gjemnessund',gje_max_batches=0,use_amp=False,
                                 model='PULSE',pred_len=12,label_len=0,features='M')
        exp.model=Zero()
        exp.device=torch.device('cpu')
        ds.weighted_evaluation=True
        self.assertAlmostEqual(exp.vali(ds,loader,torch.nn.MSELoss()),11/3,places=6)
        ds.weighted_evaluation=False
        exp.args.data='ETTh1'
        self.assertAlmostEqual(exp.vali(ds,loader,torch.nn.MSELoss()),5,places=6)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
