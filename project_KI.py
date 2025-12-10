import csv
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path
from math import prod
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import json
import random
import ast
#from torchmetrics import ConfusionMatrix


random.seed(18)


#$$$$$$$$$$$$$-----

#variables
batch_size=128
train_split_pct=0.7
validation_split_pct=0.15
test_split_pct=0.15
epoch_number=150
lr=1e-3
min_delta=1e-4

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
                         ])

#$$$$$$$$$$$$$$$$-------

print(f'Batch size: {batch_size}')

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
for i in annotations:
    path=i.get('image_filepath')
    full_path=images_folder_path/path
    answers = i.get('answers')

    relevant_value = None
    social_media_channel_value = None
    creator_value = None

    for a in answers:
        if a.get('variable')=='is-relevant':
            relevant_value = a.get('answer')
            for b in answers:
                if b.get('variable') == 'creator':
                    creator_value = b.get('answer')
                elif b.get('variable') == 'social-media-channel':
                    social_media_channel_value = b.get('answer')


    rows.append({'image_filepath': str(full_path),
                 'is-relevant': relevant_value,
                 'social-media-channel': social_media_channel_value,
                 'creator': creator_value})       #rows= πινακας με 1 λεξικό για καθε εικόνα


#δημιουργια dataframe από rows
rows_df=pd.DataFrame(rows)
yes_df=rows_df[rows_df['is-relevant']=='Yes']
new_df=yes_df[['image_filepath','social-media-channel','creator']]
print(f'Number of relevant images:',len(new_df))


#εντοπισμος unique labels και δημιουργια label maps
social_media_channel_set=set()
for y in new_df['social-media-channel']:
    social_media_channel_set.add(y)

creator_flattened_set=set()
for x in new_df['creator']:
    if isinstance(x, list):
        creator_flattened_set.update(x)  #αν είναι λίστα,προσθέτει κάθε στοιχείο της χωριστά στο flattened_list
    else:
        creator_flattened_set.add(x)

sorted_list1=sorted(social_media_channel_set)
sorted_list2=sorted(creator_flattened_set)

social_media_channel_label_map={s:i for i,s in enumerate(sorted_list1)}
creator_label_map={s:i for i,s in enumerate(sorted_list2)}

print(f'Social media label map: \n {social_media_channel_label_map}')
print(f'Creator label map: \n {creator_label_map}')



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
                 social_media_channel_label_map,
                 creator_label_map,
                 transform=None):
        self.transform=transform
        self.samples =pd.read_csv(csvfile)
        self.creator_label_map=creator_label_map
        self.social_media_channel_label_map = social_media_channel_label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, creator_index):
        sample=self.samples.iloc[creator_index]   #αποθηκευση του i-οστού λεξικου στη μεταβλητη sample
        image_path=sample['image_filepath']  #αποθηκευση του string path της εικονας στη μεταβλτητη image_path
        raw_creator_value=sample['creator']
        social_media_channel_label_value = self.social_media_channel_label_map.get(sample['social-media-channel'],None)

        if isinstance(raw_creator_value, str):
            raw_creator_value=raw_creator_value.strip()   #αφαιρει τυχον κενα απο το string σε αρχη και τελος
            if raw_creator_value.startswith('[') and raw_creator_value.endswith(']'):
                creator_labels=ast.literal_eval(raw_creator_value)  #γινεται πραγματικη python λιστα
            else:
                creator_labels=[raw_creator_value]
        elif isinstance(raw_creator_value, list):
            creator_labels=raw_creator_value
        else:
            creator_labels=[]

        num_classes=len(self.creator_label_map)
        creator_multi_hot_vector=torch.zeros(num_classes, dtype=torch.float32)

        for lbl in creator_labels:
            creator_index=self.creator_label_map.get(lbl, None)
            if creator_index is None:
                raise ValueError(f'!No creator label!')
            creator_multi_hot_vector[creator_index]=1

        image=Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image=self.transform(image)

        mean=image.mean(dim=(1,2), keepdim=True)  #(3,1,1) - υπολογιζει mean σε στις διαστασεις (1,2) δλδ (H,W)
        std=image.std(dim=(1, 2), keepdim=True)  #(3,1,1)
        std = torch.clamp(std, min=1e-6)  #για να μη διαιρεσει με 0
        image=(image-mean)/std

        return image, social_media_channel_label_value ,creator_multi_hot_vector



#δημιουργία datasets
training_dataset=ImageDataset(train_csv,social_media_channel_label_map, creator_label_map, transform=transform)
validation_dataset=ImageDataset(validation_csv,social_media_channel_label_map, creator_label_map, transform=transform)
test_dataset=ImageDataset(test_csv,social_media_channel_label_map, creator_label_map, transform=transform)

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



model=Network(input_dims=(224,224,3),output_dims=len(creator_label_map))

#χρήση GPU (εαν υπάρχει)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'running on device:',device)

model.to(device)
criterion_single_label=nn.CrossEntropyLoss()
criterion_multi_label=nn.BCEWithLogitsLoss()
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
    running_loss=0.0
    train_total=0


    for images, social_media_channel_labels, creator_labels in train_loader:
        images=images.to(device)
        social_media_channel_labels=social_media_channel_labels.to(device)
        creator_labels=creator_labels.to(device)
        optimizer.zero_grad()
        x=model(images)
        loss1=criterion_single_label(x,social_media_channel_labels)
        loss2=criterion_multi_label(x,creator_labels)
        loss=loss1+loss2
        loss.backward()
        optimizer.step()
        running_loss+= loss.item() * images.size(0)

        train_total += labels.size(0)

    epoch_loss= running_loss / len(train_loader.dataset)
    print(f'Training loss: {epoch_loss}')


#validation
    with torch.no_grad():

        model.eval()
        validation_loss = 0.0
        val_correct = 0
        val_total = 0
        TP_val = torch.zeros(len(creator_label_map), dtype=torch.long)
        TN_val = torch.zeros(len(creator_label_map), dtype=torch.long)
        FP_val = torch.zeros(len(creator_label_map), dtype=torch.long)
        FN_val = torch.zeros(len(creator_label_map), dtype=torch.long)

        for images, social_media_channel_labels, creator_labels in validation_loader:
            images=images.to(device)
            social_media_channel_labels = social_media_channel_labels.to(device)
            creator_labels = creator_labels.to(device)
            x = model(images)
            loss1=criterion_single_label(x,social_media_channel_labels)
            loss2 = criterion_multi_label(x, creator_labels)
            loss=loss1+loss2
            validation_loss+= loss.item() * images.size(0)
            probs=torch.sigmoid(x)    #κανει τα logits->πιθανοτητες
            preds_val = (probs>0.5).float()

            TP_batch = ((preds_val == 1) & (creator_labels == 1)).sum(dim=0)  # μετράει τα True σε καθε στηλη
            TN_batch = ((preds_val == 0) & (creator_labels == 0)).sum(dim=0)
            FP_batch = ((preds_val == 1) & (creator_labels == 0)).sum(dim=0)
            FN_batch = ((preds_val == 0) & (creator_labels == 1)).sum(dim=0)

            TP_val += TP_batch.cpu()
            TN_val += TN_batch.cpu()
            FP_val += FP_batch.cpu()
            FN_val += FN_batch.cpu()

        final_val_loss = validation_loss / len(validation_loader.dataset)
        total_creator_label=TP_val+TN_val+FP_val+FN_val
        validation_accuracy=(TP_val+TN_val)/total_creator_label

        macro_validation_accuracy=validation_accuracy.mean().item()


    #print(f'TP: {TP_val},TN: {TN_val},FP: {FP_val},FN: {FN_val}')
    #print(f'Total creator label: {total_creator_label}')
    #print(f'Validation accuracy: {validation_accuracy}')


    writer.add_scalars('Accuracy Metrics', {
         'Validation Accuracy': macro_validation_accuracy,
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
    all_preds=[]
    TP_testing = torch.zeros(len(creator_label_map), dtype=torch.long)
    TN_testing = torch.zeros(len(creator_label_map), dtype=torch.long)
    FP_testing = torch.zeros(len(creator_label_map), dtype=torch.long)
    FN_testing = torch.zeros(len(creator_label_map), dtype=torch.long)

    for images, social_media_channel_labels, creator_labels in test_loader:
        images=images.to(device)
        social_media_channel_labels = social_media_channel_labels.to(device)
        creator_labels = creator_labels.to(device)
        x=model(images)
        loss1=criterion_single_label(x,social_media_channel_labels)
        loss2=criterion_multi_label(x, creator_labels)
        loss=loss1+loss2
        testing_loss+= loss.item() * images.size(0)
        probs=torch.sigmoid(x)
        preds_testing = (probs > 0.5)

        TP_batch = ((preds_testing == 1) & (creator_labels == 1)).sum(dim=0)  # μετράει τα True σε καθε στηλη
        TN_batch = ((preds_testing == 0) & (creator_labels == 0)).sum(dim=0)
        FP_batch = ((preds_testing == 1) & (creator_labels == 0)).sum(dim=0)
        FN_batch = ((preds_testing == 0) & (creator_labels == 1)).sum(dim=0)

        TP_testing += TP_batch.cpu()
        TN_testing += TN_batch.cpu()
        FP_testing += FP_batch.cpu()
        FN_testing += FN_batch.cpu()
        total_creator_label = TP_testing + TN_testing + FP_testing + FN_testing


        all_preds.extend(preds_testing.cpu().numpy().astype(int))
        all_labels.extend(creator_labels.cpu().numpy().astype(int))


        testing_accuracy=(TP_testing+TN_testing)/total_creator_label #per label accuracy
        macro_testing_accuracy=testing_accuracy.mean().item()*100  #συνολικο accuracy
        testing_f1=f1_score(all_labels,all_preds,average=None,zero_division=0)
        testing_recall=recall_score(all_labels,all_preds,average=None,zero_division=0)
        testing_precision=precision_score(all_labels,all_preds,average=None, zero_division=0)


    final_test_loss=testing_loss/len(test_loader.dataset)
    #conf_matrix=ConfusionMatrix(num_classes=7)
    #conf_matrix=confusion_matrix(all_labels, all_predictions, labels=np.arange(len(creator_label_map)))
    print(f'TP: {TP_testing},\n TN: {TN_testing},\n FP: {FP_testing},\n FN: {FN_testing}')
    print(f'->Testing Accuracy: \n {macro_testing_accuracy:.3f}% \n->Testing Loss:\n {final_test_loss:.5f}')
    #print(conf_matrix)
    print(f'F1 score: {testing_f1}')
    print(f'Recall score: {testing_recall}')
    print(f'Precision score: {testing_precision}')

#plt.imshow(conf_matrix)
#plt.show()

print('end')
if __name__ == '__main__':
    pass
