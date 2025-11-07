import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights
import random
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import json
import csv

#αποθηκευση των runs σε υποφακελους
day_stamp=datetime.datetime.now().strftime("%Y-%m-%d")
time_stamp=datetime.datetime.now().strftime("%H-%M-%S")
day_path= Path('runs/tb_logs') / day_stamp
time_path=day_path/time_stamp
day_path.mkdir(parents=True, exist_ok=True)
time_path.mkdir(parents=True, exist_ok=True)
writer=SummaryWriter(time_path)


os.makedirs('saved_models', exist_ok=True)


#variables
batch_size=64
train_split_pct=0.7
validation_split_pct=0.15
test_split_pct=0.15
epoch_number=200
lr=1e-3
min_delta=1e-4

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,64)),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                         ])


social_media_label_map={'Instagram':0,
                        'Pinterest':1,
                        'Snapchat':2,
                        'TikTok':3,
                        'YouTube':4,
                        'Other':5,
                        'Not sure':6}


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
for i in annotations[:6000]:
    path=i.get('image_filepath')
    full_path=images_folder_path/path
    answers = i.get('answers')

    relevant_value = None
    social_media_value = None

    for a in answers:
        if a.get('variable')=='is-relevant' and a.get('answer')=='Yes':
            relevant_value = a.get('answer')
            for b in answers:
                if b.get('variable') == 'social-media-channel':
                    social_media_value = b.get('answer')

    rows.append({'image_filepath': str(full_path),'is-relevant': str(relevant_value), 'social-media-channel': str(social_media_value)})
    #rows= πινακας με 1 λεξικό για καθε εικόνα



#ανοιγμα csv file και προσθηκη στηλών
filtered_rows=[r for r in rows if r.get('is-relevant')=='Yes']
csv_fieldnames=('image_filepath','social-media-channel')

with open('project_ki/csv_files/CSV_edited.csv', 'w', newline='') as csvfile:
    csv_writer=csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
    csv_writer.writeheader()   #γραφει τα ονοματα των στηλών
    for row in filtered_rows:
            csv_writer.writerow({
                'image_filepath': row['image_filepath'],
                'social-media-channel': row['social-media-channel'],
                    })


print(f'Length of filtered rows',len(filtered_rows))


#split csv
csv_length=len(filtered_rows)
train_split=int(np.ceil(train_split_pct * csv_length))  #70% του συνόλου του rows
validation_split=int(np.ceil(validation_split_pct * csv_length))
test_split=int(csv_length - validation_split - train_split)

random.shuffle(filtered_rows)
train_rows=filtered_rows[:train_split]
validation_rows=filtered_rows[train_split:train_split+validation_split]
test_rows=filtered_rows[train_split+validation_split:]

print(f'Length of train rows',len(train_rows))
print(f'Length of validation rows',len(validation_rows))
print(f'Length of test rows',len(test_rows))

csv_fieldnames=('image_filepath','social-media-channel')


#συναρτηση δημιουργιας csv splits
def csv_creator(filename,data):
    with open(filename,'w',newline='') as f:
        creator=csv.DictWriter(f,fieldnames=csv_fieldnames)
        creator.writeheader()
        for k in data:
            creator.writerow({
                'image_filepath': k['image_filepath'],
                'social-media-channel': k['social-media-channel']
                    })

        return filename

train_csv=csv_creator('project_ki/csv_files/train_csv.csv', train_rows)
validation_csv=csv_creator('project_ki/csv_files/validation_csv.csv', validation_rows)
test_csv=csv_creator('project_ki/csv_files/test_csv.csv', test_rows)


#κλάση δημιουργίας dataset

class ImageDataset(Dataset):
    def __init__(self,
                 csvfile,
                 social_media_label_map,
                 transform=None):
        self.transform=transform
        self.samples = []
        self.social_media_label_map=social_media_label_map
        with open(csvfile, 'r', newline='') as data_file:
            reader=csv.DictReader(data_file)
            for i in reader:
                self.samples.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample=self.samples[index]   #αποθηκευση του i-οστού λεξικου στη μεταβλητη sample
        image_path=sample['image_filepath']     #αποθηκευση του string path της εικονας στη μεταβλτητη image_path
        social_media_label_value=self.social_media_label_map.get(sample['social-media-channel'])     #αναζητηση της τιμης που αντιστοιχει στο συγκ. κλειδι μέσα στο label map, αν δεν βρει κατι βαζει 0

        image=Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        return image, social_media_label_value  #επιστρεφει tuple ((3,224,224),0/1)


#δημιουργία datasets
training_dataset=ImageDataset(train_csv,social_media_label_map,transform=transform)
validation_dataset=ImageDataset(validation_csv,social_media_label_map,transform=transform)
test_dataset=ImageDataset(test_csv,social_media_label_map,transform=transform)

train_loader=DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
validation_loader=DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


#-----------------------------------------#NETWORK---------------------------------------

#χρήση GPU (εαν υπάρχει)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#φορτωνω pretrained resnet18
weights=ResNet18_Weights.DEFAULT
model=resnet18(weights=weights).to(device)

model.fc=nn.Linear(model.fc.in_features, 7)    #classifier

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



    #torch.save(model, f'saved_models/model_epoch_{epoch}.pth')   #αποθηκευση μοντέλου καθε epoch ως ξεχωριστο αρχειο
    writer.add_scalar('Training Loss', epoch_loss, epoch)



#validation
    with torch.no_grad():
        model.eval()
        validation_loss = 0.0
        val_correct = 0
        val_total = 0

        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            x = model(images)
            loss = criterion(x, labels)
            validation_loss+=loss.item()*images.size(0)
            preds = x.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += images.size(0)

        val_acc = (val_correct /val_total)*100
        final_val_loss = validation_loss / len(validation_loader.dataset)

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
        images = images.to(device)
        labels = labels.to(device)
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
