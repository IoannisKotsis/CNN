import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import Subset
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights


#αποθηκευση των runs σε υποφακελους
day_stamp=datetime.datetime.now().strftime("%Y-%m-%d")
time_stamp=datetime.datetime.now().strftime("%H-%M-%S")
day_path=Path('runs/logs')/day_stamp
time_path=day_path/time_stamp
day_path.mkdir(parents=True, exist_ok=True)
time_path.mkdir(parents=True, exist_ok=True)
writer=SummaryWriter(time_path)


os.makedirs('saved_models', exist_ok=True)


transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

full_ds=datasets.MNIST(root='mnist',train=True,transform=transform,download=True)  #φορτωνει το MNIST train_and_val (60.000 εικονες)
train_ds_full=datasets.MNIST(root='mnist',train=False,transform=transform,download=True)

#g = torch.Generator().manual_seed(0)
train_ds=Subset(full_ds, range(5500))
val_ds=Subset(full_ds, range(5500,7500))
test_ds=Subset(train_ds_full, range(3500))
#train_ds, val_ds, test_ds = random_split(full_ds,[50000,5000,5000], generator=g)


#variables
batch_size=600
epoch_number=10
lr=1e-3
min_delta=1e-4

train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True)  #πακετάρισμα training δεδομενων σε batches των 64
val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False)   #πακετάρισμα validation δεδομενων σε batches των 256
test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False)  #πακετάρισμα test δεδομενων σε batches των 128


#-----------------------------------------#NETWORK---------------------------------------


#φορτωνω pretrained resnet18
weights=ResNet18_Weights.DEFAULT
model=resnet18(weights=weights)

model.fc=nn.Linear(model.fc.in_features, 10)    #classifier

#παγωνω βάρη του μοντελου
for p in model.parameters():
    p.requires_grad=False


#ξεπαγωνω layers
for name, p in model.named_parameters():
    if name.startswith(('layer4','fc')):
        p.requires_grad=True


trainable_params=(p for p in model.parameters() if p.requires_grad)
optimizer=optim.Adam(trainable_params,lr=lr)

optimizer=optim.Adam([
    {'params': [p for name,p in model.named_parameters() if name.startswith(('layer4'))], 'lr': 1e-4},
    {'params': [p for name,p in model.named_parameters() if name.startswith(('fc'))], 'lr': 1e-5},
])

criterion=nn.CrossEntropyLoss()


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
    print(f'->Testing Accuracy: \n {test_acc:.2f}% \n->Testing Loss:\n {final_test_loss:.5f}')


if __name__ == '__main__':
    pass
