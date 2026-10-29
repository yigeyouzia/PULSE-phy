"""Source copy from FACT-main/models/FACT.py.

Computation unchanged; adaptations live outside vendor/. See sources.json and licenses.
"""

import torch
import torch.nn as nn
import math


class Inception2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, dilation=1, groups=1, padding_mode='zeros', init_weight=True):
        super(Inception2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.dilation = dilation
        self.kernels = nn.ModuleList([
                            nn.Conv2d(in_channels, out_channels, kernel_size=2 * (i+1) + 1, 
                                dilation=dilation, padding=dilation * (i+1), 
                                padding_mode=padding_mode, groups=groups) 
                            for i in range(self.num_kernels-1)])
                             
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                dilation=1, padding=0, 
                                padding_mode=padding_mode, groups=groups)
       
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        tmp = self.conv1(x)
        for i in range(self.num_kernels-1):
            tmp = tmp + self.kernels[i](x) 
        return tmp/self.num_kernels


class Conv_block2d(nn.Module):
    def __init__(self, configs, dilation=3, d_model=512, d_ff=2048, head=1):
        super(Conv_block2d, self).__init__()
        if configs.freq == 't':
            self.new_channel = configs.enc_in+5 
        elif configs.freq == 'h':
            self.new_channel = configs.enc_in+4
        elif configs.freq == 'd':
            self.new_channel = configs.enc_in+3
        else:
            self.new_channel = configs.enc_in
        self.channel_patch = int(math.sqrt(self.new_channel)) +1
        self.channel_patch2 = self.channel_patch
        
        self.d_model = d_model
        self.head = head
        self.dilation = dilation
        self.num_kernel = configs.num_kernels

        self.conv = nn.ModuleList([  nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            Inception2d(self.d_model, self.d_model, num_kernels=self.num_kernel, 
                dilation=d+1, padding_mode='zeros', groups=self.d_model),
            )     for d in range(self.dilation)])


        self.conv_ffn = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=1,stride=1,padding=0,groups=1),
            torch.nn.GELU( ),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1) 
            )   


    def forward(self, x):
        batch, seq, channel = x.shape
        x = torch.cat((x, x[:,:,:(self.channel_patch*self.channel_patch2-channel)]),dim=-1)
        x = x.reshape(batch, seq, self.channel_patch, self.channel_patch2)

        time =  x
        for i in range(self.dilation):
            time = time + self.conv[i](x)
        x = time

        x = x + self.conv_ffn(x)
        x = x.reshape(batch, seq, -1)[:, :, :channel]
        return x
