import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sympy.physics.control.control_plots import matplotlib
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import Subset
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path
from math import prod
from torchmetrics import ConfusionMatrix
from matplotlib import pyplot as plt

#torch.manual_seed(15)

#αποθηκευση των runs σε υποφακελους
day_stamp=datetime.datetime.now().strftime("%Y-%m-%d")
time_stamp=datetime.datetime.now().strftime("%H-%M-%S")
day_path=Path('runs/logs')/day_stamp
time_path=day_path/time_stamp
day_path.mkdir(parents=True, exist_ok=True)
time_path.mkdir(parents=True, exist_ok=True)
writer=SummaryWriter(time_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs('saved_models', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

transform=transforms.Compose([transforms.Resize(size=(28,28)),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])     #μετατροπη εικονων σε PyTorch float tensor + κανονικοποιηση τιμων + κανονικοποιηση

full_ds=datasets.MNIST(root='mnist',train=True,transform=transform,download=True)  #φορτωνει το MNIST train_and_val (60.000 εικονες)
train_ds_full=datasets.MNIST(root='mnist',train=False,transform=transform,download=True)

#g = torch.Generator().manual_seed(0)
train_ds=Subset(full_ds, range(5500))
val_ds=Subset(full_ds, range(5500,7500))
test_ds=Subset(train_ds_full, range(3500))
#train_ds, val_ds, test_ds = random_split(full_ds,[50000,5000,5000], generator=g)


#variables
batch_size=16
epoch_number=6
lr=1e-3
min_delta=1e-4

train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True)  #πακετάρισμα training δεδομενων σε batches των 64
val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False)   #πακετάρισμα validation δεδομενων σε batches των 256
test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False)  #πακετάρισμα test δεδομενων σε batches των 128


#-----------------------------------------#NETWORK---------------------------------------


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

        #Περιπτώσεις padding
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
        x=F.gelu(x)
        x=self.pool(x)
        return x

    def output_dims(self):
        combined_tuples=(*self._output_shape,self._output_channels)
        return combined_tuples


class Network(nn.Module):
    def __init__(self,
                 input_dims,
                 output_dims,
                 linear1_output_size=100,
                 linear2_output_size=100
                 ):
        super().__init__()
        self.block1=ConvBlock(input_dims, 8)                     #28->14
        self.block2=ConvBlock(self.block1.output_dims(), 16)     #14->7   δινει (8,(14,14))
        self.block3=ConvBlock(self.block2.output_dims(), 32)     #7->3


        self.flatten = nn.Flatten()
        self.gelu=nn.GELU()
        self.dropout=nn.Dropout(0.4)
        self.fc1=nn.Linear(prod(self.block3.output_dims()) , linear1_output_size)
        self.fc2 = nn.Linear(linear1_output_size, linear2_output_size)
        self.output_layer = nn.Linear(linear2_output_size, output_dims)


    def forward(self, x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.flatten(x)
        x=self.gelu(self.fc1(x))
        x=self.dropout(x)
        x=self.gelu(self.fc2(x))
        x=self.dropout(x)
        x=self.output_layer(x)
        return x


model=Network(input_dims=(28,28,1),output_dims=10)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr)



#καθορισμος παραμετρων early stopping
import copy
best_val_loss=float('inf')
best_state=None
patience=5
wait=0


#training
for epoch in range(epoch_number):
    model.train()
    running_loss = 0.0
    train_correct=0
    train_total=0

    for images, labels in train_loader:
        optimizer.zero_grad()
        x=model(images)
        loss=criterion(x, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*images.size(0)
        preds_train=x.argmax(1)
        train_correct+=(preds_train==labels).sum().item()
        train_total+=images.size(0)
    epoch_loss=running_loss/len(train_loader.dataset)
    training_accuracy=(train_correct/train_total)*100



    #torch.save(model, f'saved_models/model_epoch_{epoch}.pth')   #αποθηκευση μοντέλου καθε epoch ως ξεχωριστο αρχειο
    writer.add_scalar('Training Loss', epoch_loss, epoch)

#validation
    with torch.no_grad():
        model.eval()
        validation_loss = 0.0
        val_correct = 0
        val_total = 0

        for images, labels in val_loader:
            x = model(images)
            loss = criterion(x, labels)
            validation_loss+=loss.item()*images.size(0)
            preds = x.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += images.size(0)

        val_acc = (val_correct /val_total)*100
        final_val_loss = validation_loss / len(val_loader.dataset)

    #print(f'--Epoch {epoch+1} has loss: {epoch_loss:.6f} \n  Validation Loss {epoch+1}: {final_val_loss:.6f} \n  Validation Accuracy: {val_acc:.2f}%')
    writer.add_scalar('Validation Loss', final_val_loss, epoch)
    writer.add_scalars('Accuracy Metrics', {
        'Validation Accuracy': val_acc,
        'Training Accuracy': training_accuracy
    }, epoch)


#early stopping
    if (best_val_loss-final_val_loss)>min_delta:
        best_val_loss=final_val_loss
        best_state=copy.deepcopy(model.state_dict())
        torch.save({                                               #αποθηκευση του λεξικού με το καλύτερο μοντέλο σε pytorch αρχείο
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss': epoch_loss
            },'checkpoints/CNN_best_model.pth')


        wait=0
    else:
        wait+=1
        if wait>=patience:
            print(f'----Training stopped after {epoch} epochs with best validation loss: {best_val_loss:.6f}----')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

writer.close()

#testing
model.eval()
with torch.no_grad():
    testing_loss=0.0
    test_correct=0
    test_total=0

    for images, labels in test_loader:
        x=model(images)
        loss=criterion(x, labels)
        testing_loss+=loss.item()*images.size(0)
        predictions=x.argmax(1)
        test_correct+=(predictions==labels).sum().item()
        test_total+=images.size(0)
    test_acc=(test_correct/test_total)*100
    final_test_loss=testing_loss/len(test_loader.dataset)
    confmat=ConfusionMatrix(task='multiclass',num_classes=10)
    conf_matrix=confmat(predictions,labels)
    print(f'->Testing Accuracy: \n {test_acc:.2f}% \n->Testing Loss:\n {final_test_loss:.5f}')
    print(conf_matrix)

#plt.imshow(conf_matrix)
#plt.show()


if __name__ == '__main__':
    pass
