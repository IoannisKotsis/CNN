import csv
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path
from math import prod
from sklearn.metrics import confusion_matrix
import json
import random
#from torchmetrics import ConfusionMatrix
random.seed(18)

#$$$$$$$$$$$$$-----

#variables
batch_size=64
train_split_pct=0.7
validation_split_pct=0.15
test_split_pct=0.15
epoch_number=150
lr=1e-3
min_delta=1e-4

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((354,768))
                         ])


social_media_label_map={'Instagram':0,
                        'Pinterest':1,
                        'Snapchat':2,
                        'TikTok':3,
                        'YouTube':4,
                        'Other':5,
                        'Not sure':6}

#$$$$$$$$$$$$$$$$-------

#δημιουργία φακέλου για tensorboard
day_stamp=datetime.datetime.now().strftime("%Y-%m-%d")
time_stamp=datetime.datetime.now().strftime("%H-%M-%S")
day_path= Path('runs/tb_logs') / day_stamp
time_path=day_path/time_stamp
day_path.mkdir(parents=True, exist_ok=True)
time_path.mkdir(parents=True, exist_ok=True)
writer=SummaryWriter(time_path)

#φάκελος του project
project_ki='project_ki'
os.makedirs('project_ki', exist_ok=True)
os.makedirs(os.path.join('project_ki','data'),exist_ok=True)
os.makedirs(os.path.join('project_ki','csv_files'),exist_ok=True)
os.makedirs(os.path.join(project_ki,'checkpoints'),exist_ok=True)

#ανοιγμα του json file
with open('/home/ioankots/projects/CNN/datasets/digital-ads/image-annotations.questionnaire_answers.json', 'r', encoding='utf-8') as file:
    annotations=json.load(file)

#path των εικόνων
images_folder_path=Path("/home/ioankots/projects/CNN/datasets/digital-ads")


rows=[]
#επιλογη των paths και των values που θελω
for i in annotations[:2000]:
    path=i.get('image_filepath')
    full_path=images_folder_path/path
    answers = i.get('answers')

    relevant_value = None
    social_media_value = None

    for a in answers:
        if a.get('variable')=='is-relevant':
            relevant_value = a.get('answer')
            for b in answers:
                if b.get('variable') == 'social-media-channel':
                    social_media_value = b.get('answer')

    rows.append({'image_filepath': str(full_path),'is-relevant': str(relevant_value), 'social-media-channel': str(social_media_value)})
    #rows= πινακας με 1 λεξικό για καθε εικόνα


#δημιουργια dataframe από rows
rows_df=pd.DataFrame(rows)
yes_df=rows_df[rows_df['is-relevant']=='Yes']
new_df=yes_df[['image_filepath','social-media-channel']]

print(f'Length of filtered rows',len(new_df))


#split dataframe
df_length=len(new_df)
train_split=int(train_split_pct * df_length)
validation_split=int(validation_split_pct * df_length)

#δημιουργια των splits
shuffled_df=new_df.sample(frac=1,random_state=18)
train_rows=shuffled_df[:train_split]
validation_rows=shuffled_df[train_split:train_split+validation_split]
test_rows=shuffled_df[train_split+validation_split:]

#δημιουργια των csv files
train_rows.to_csv('project_ki/csv_files/train_csv.csv', index=False)
validation_rows.to_csv('project_ki/csv_files/validation_csv.csv', index=False)
test_rows.to_csv('project_ki/csv_files/test_csv.csv', index=False)

train_csv='project_ki/csv_files/train_csv.csv'
validation_csv='project_ki/csv_files/validation_csv.csv'
test_csv='project_ki/csv_files/test_csv.csv'


#κλάση δημιουργίας dataset

class ImageDataset(Dataset):
    def __init__(self,
                 csvfile,
                 social_media_label_map,
                 transform=None):
        self.transform=transform
        self.samples =pd.read_csv(csvfile)
        self.social_media_label_map=social_media_label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample=self.samples.iloc[index]   #αποθηκευση του i-οστού λεξικου στη μεταβλητη sample
        image_path=sample['image_filepath']     #αποθηκευση του string path της εικονας στη μεταβλτητη image_path
        social_media_label_value=self.social_media_label_map.get(sample['social-media-channel'])     #αναζητηση της τιμης που αντιστοιχει στο συγκ. κλειδι μέσα στο label map, αν δεν βρει κατι βαζει 0
        image=Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image=self.transform(image)

        mean=image.mean(dim=(1,2), keepdim=True)  #(3,1,1)
        std=image.mean(dim=(1, 2), keepdim=True)  #(3,1,1)
        std = torch.clamp(std, min=1e-6)  #για να μη διαιρεσει με 0
        image=(image-mean)/std

        return image, social_media_label_value  #επιστρεφει tuple ((3,224,224),0,1,...6)


#δημιουργία datasets
training_dataset=ImageDataset(train_csv,social_media_label_map,transform=transform)
validation_dataset=ImageDataset(validation_csv,social_media_label_map,transform=transform)
test_dataset=ImageDataset(test_csv,social_media_label_map,transform=transform)

train_loader=DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
validation_loader=DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

print(f'Length of training dataset',len(training_dataset))
print(f'Length of validation dataset',len(validation_dataset))
print(f'Length of test dataset',len(test_dataset))



print(f'Train loader size',len(train_loader))
print(f'Validation loader size',len(validation_loader))
print(f'Test loader size',len(test_loader))



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
        χ=self.bn1(x)
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
                 linear1_output_size=50,
                 linear2_output_size=50
                 ):
        super().__init__()
        self.block1=ConvBlock(input_dims, 32)  #βγαζει 8 feature maps
        self.block2=ConvBlock(self.block1.output_dims(), 64)
        self.block3=ConvBlock(self.block2.output_dims(), 128)


        self.flatten = nn.Flatten()
        self.gelu=nn.GELU()
        self.dropout=nn.Dropout(0.2)
        self.fc1=nn.Linear(prod(self.block3.output_dims()) , linear1_output_size)
        self.fc2 = nn.Linear(linear1_output_size, linear2_output_size)
        self.output_layer = nn.Linear(linear2_output_size, output_dims)


    def forward(self, x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.flatten(x)
        x=self.gelu(self.fc1(x))
        x=self.gelu(self.fc2(x))
        x=self.dropout(x)
        x=self.output_layer(x)
        return x



model=Network(input_dims=(64,64,3),output_dims=7)

#χρήση GPU (εαν υπάρχει)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'running on device:',device)

model.to(device)
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
        images=images.to(device)
        labels=labels.to(device)
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



#validation
    with torch.no_grad():

        model.eval()
        validation_loss = 0.0
        val_correct = 0
        val_total = 0

        for images, labels in validation_loader:
            images=images.to(device)
            labels=labels.to(device)
            x = model(images)
            loss = criterion(x, labels)
            validation_loss+=loss.item()*images.size(0)
            preds = x.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += images.size(0)

        val_acc = (val_correct/val_total)*100
        final_val_loss = validation_loss / len(validation_loader.dataset)


    #print(f'--Epoch {epoch+1} has loss: {epoch_loss:.6f} \n  Validation Loss {epoch+1}: {final_val_loss:.6f} \n  Validation Accuracy: {val_acc:.2f}%')


    writer.add_scalars('Accuracy Metrics', {
        'Training Accuracy': training_accuracy,
        'Validation Accuracy': val_acc,
    }, epoch)

    writer.add_scalars('Loss Curves', {
        'Training Loss': epoch_loss,
        'validation Loss': final_val_loss
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
            },'project_ki/checkpoints/CNN_best_model.pth')


        wait=0
    else:
        wait+=1
        if wait>=patience:
            print(f'->Training stopped after {epoch} epochs with best validation loss: {best_val_loss:.6f}')
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
    all_labels=[]
    all_predictions=[]

    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        x=model(images)
        loss=criterion(x, labels)
        testing_loss+=loss.item()*images.size(0)
        predictions=x.argmax(1)
        test_correct+=(predictions==labels).sum().item()
        test_total+=images.size(0)
        all_labels.extend(labels)
        all_predictions.extend(predictions)
    test_acc=(test_correct/test_total)*100
    final_test_loss=testing_loss/len(test_loader.dataset)
    #conf_matrix=ConfusionMatrix(num_classes=7)
    conf_matrix=confusion_matrix(labels.cpu().numpy().astype(int),predictions.cpu().numpy().astype(int),labels=np.arange(7))
    print(f'->Testing Accuracy: \n {test_acc:.2f}% \n->Testing Loss:\n {final_test_loss:.5f}')
    print(conf_matrix)

#plt.imshow(conf_matrix)
#plt.show()

print('end')
if __name__ == '__main__':
    pass
