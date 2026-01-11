import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self,
                 input_dims,
                 num_filters,
                 conv_kernel_size=3,
                 conv_stride=1,
                 padding=1,
                 pool_kernel_size=2,
                 pool_stride=2,
                 pool_padding=0,
                 ):
        super().__init__()
        self._input_shape = input_dims[:-1]       #κρατάει (28,28) απο το (28,28,1)
        self._input_channels = input_dims[-1]   #κρατάει (1) απο το (28,28,1)

        #padding cases
        if padding == 'same':
            self._output_shape = np.floor(((np.ceil(np.asarray(self._input_shape) / np.asarray(conv_stride)) + 2 * np.asarray(pool_padding) - np.asarray(pool_kernel_size)) / np.asarray(pool_stride)) + 1).astype(int)
        elif padding == 'valid':
            self.cr = np.floor(((np.asarray(self._input_shape) - np.asarray(conv_kernel_size)) / np.asarray(conv_stride)) + 1).astype(int)
            self._output_shape = np.floor(((self.cr + 2 * np.asarray(pool_padding) - np.asarray(pool_kernel_size)) / np.asarray(pool_stride)) + 1).astype(int)
        elif isinstance(padding, int):
            self.cr=np.floor(((np.asarray(self._input_shape)+2*padding-np.asarray(conv_kernel_size))/np.asarray(conv_stride))+1).astype(int)
            self._output_shape = np.floor(((self.cr + 2 * np.asarray(pool_padding) - np.asarray(pool_kernel_size)) / np.asarray(pool_stride)) + 1).astype(int)
        else:
            raise NotImplementedError

        self._output_channels = num_filters
        self.conv=nn.Conv2d(input_dims[-1], num_filters, kernel_size=conv_kernel_size, stride=conv_stride, padding=padding)
        self.bn1=nn.BatchNorm2d(num_filters)
        self.pool=nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn1(x)
        x=F.gelu(x)
        x=self.pool(x)
        return x

    def output_dims(self):
        combined_tuples=(*self._output_shape,self._output_channels)
        return combined_tuples


class Network(nn.Module):
    def __init__(self,
                 input_dims,
                 output_dims_single_label,
                 output_dims_multi_label,
                 output_dims_binary_label,
                 linear1_output_size=64,
                 linear2_output_size=64
                 ):
        super().__init__()
        self.block1=ConvBlock(input_dims, 32)  #βγαζει 32 feature maps
        self.block2=ConvBlock(self.block1.output_dims(), 64)
        self.block3=ConvBlock(self.block2.output_dims(), 128)


        self.flatten = nn.Flatten()
        self.gelu=nn.GELU()
        self.dropout=nn.Dropout(0.2)
        self.fc1=nn.Linear(prod(self.block3.output_dims()) , linear1_output_size)
        self.fc2 = nn.Linear(linear1_output_size, linear2_output_size)
        self.output_layer1 = nn.Linear(linear2_output_size, output_dims_single_label)
        self.output_layer2 = nn.Linear(linear2_output_size, output_dims_multi_label)
        self.output_layer3 = nn.Linear(linear2_output_size, output_dims_binary_label)


    def forward(self, x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.flatten(x)
        x=self.gelu(self.fc1(x))
        x=self.gelu(self.fc2(x))
        x=self.dropout(x)
        single_label_head=self.output_layer1(x)
        multi_label_head= self.output_layer2(x)
        binary_label_head=self.output_layer3(x)

        return single_label_head, multi_label_head, binary_label_head

def build_model(label_maps, input_dims=(512, 512, 3)):
    num_single = len(label_maps["social"])
    num_multi = len(label_maps["creator"])
    num_binary = 1

    model = Network(
        input_dims=input_dims,
        output_dims_single_label=num_single,
        output_dims_multi_label=num_multi,
        output_dims_binary_label=num_binary
    )
    return model
