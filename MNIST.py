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



#αποθηκευση των runs σε υποφακελους
day_stamp=datetime.datetime.now().strftime("%d-%m-%Y")
time_stamp=datetime.datetime.now().strftime("%H-%M-%S")
day_path= Path('runs/tb_logs') / day_stamp
time_path=day_path/time_stamp
day_path.mkdir(parents=True, exist_ok=True)
time_path.mkdir(parents=True, exist_ok=True)
writer=SummaryWriter(time_path)


os.makedirs('saved_models', exist_ok=True)

transform=transforms.ToTensor()    #μετατροπη εικονων σε PyTorch float tensor + κανονικοποιηση τιμων

full_ds=datasets.MNIST(root='mnist',train=True,transform=transform,download=True)  #φορτωνει το MNIST train_and_val (60.000 εικονες)



train_ds=Subset(full_ds, range(5000))
val_ds=Subset(full_ds, range(5000,7000))
test_ds=Subset(full_ds, range(7000,10000))
#train_ds, val_ds, test_ds = random_split(full_ds,[50000,3500,6500])

train_loader=DataLoader(train_ds,batch_size=128,shuffle=True)  #πακετάρισμα training δεδομενων σε batches των 64
val_loader=DataLoader(val_ds,batch_size=256,shuffle=False)   #πακετάρισμα validation δεδομενων σε batches των 256
test_loader=DataLoader(test_ds, batch_size=128, shuffle=False)  #πακετάρισμα test δεδομενων σε batches των 128


input_size_l1=784
output_size_l1=128
input_size_l2=128
output_size_l2=10
epoch_number=400
lr=0.01
min_delta=1e-4

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.layer1=nn.Linear(input_size_l1,output_size_l1)
        self.layer2=nn.Linear(input_size_l2,output_size_l2)

    def forward(self, x):
        flattened=self.flatten(x)
        output1=F.relu(self.layer1(flattened))
        output2=self.layer2(output1)
        return output2

model=Network()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr)


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

    for images, labels in train_loader:
        optimizer.zero_grad()
        output2=model(images)
        loss=criterion(output2,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*images.size(0)
    epoch_loss=running_loss/len(train_loader.dataset)


    #torch.save(model, f'saved_models/model_epoch_{epoch}.pth')   #αποθηκευση μοντέλου καθε epoch ως ξεχωριστο αρχειο
    writer.add_scalar('Training Loss', epoch_loss, epoch)

#validation
    with torch.no_grad():
        model.eval()
        validation_loss = 0.0
        val_correct = 0
        val_total = 0

        for images, labels in val_loader:
            output2 = model(images)
            loss = criterion(output2, labels)
            validation_loss+=loss.item()*images.size(0)
            preds = output2.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += images.size(0)

        val_acc = (val_correct /val_total)*100
        final_val_loss = validation_loss / len(val_loader.dataset)

    #print(f'--Epoch {epoch+1} has loss: {epoch_loss:.6f} \n  Validation Loss {epoch+1}: {final_val_loss:.6f} \n  Validation Accuracy: {val_acc:.2f}%')
    writer.add_scalar('Validation Loss', final_val_loss, epoch)

#early stopping
    if (best_val_loss-final_val_loss)>min_delta:
        best_val_loss=final_val_loss
        best_state=copy.deepcopy(model.state_dict())
        wait=0
    else:
        wait+=1
        if wait>=patience:
            print(f'----Training stopped after {epoch+1} epochs with best validation loss: {best_val_loss:.6f}----')
            break

#testing
model.eval()
with torch.no_grad():
    testing_loss=0.0
    test_correct=0
    test_total=0

    for images, labels in test_loader:
        output2=model(images)
        loss=criterion(output2,labels)
        testing_loss+=loss.item()*images.size(0)
        predictions=output2.argmax(1)
        test_correct+=(predictions==labels).sum().item()
        test_total+=images.size(0)
    test_acc=(test_correct/test_total)*100
    final_test_loss=testing_loss/len(test_loader.dataset)
    print(f'->Testing Accuracy: \n {test_acc:.2f}% \n->Testing Loss:\n {final_test_loss:.5f}')



if __name__ == '__main__':
    pass
