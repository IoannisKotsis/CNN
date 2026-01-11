import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import ast
from torchvision import transforms


#dataset creation class
class ImageDataset(Dataset):
    def __init__(self,
                 csvfile,
                 social_media_channel_label_map,
                 creator_label_map,
                 logo_label_map,
                 transform=None):
        self.transform=transform
        self.samples =pd.read_csv(csvfile)
        self.creator_label_map=creator_label_map
        self.logo_label_map=logo_label_map
        self.social_media_channel_label_map = social_media_channel_label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, creator_index):
        sample=self.samples.iloc[creator_index]   #αποθηκευση του i-οστού λεξικου στη μεταβλητη sample
        image_path=sample['image_filepath']  #αποθηκευση του string path της εικονας στη μεταβλτητη image_path
        social_media_channel_label_value = self.social_media_channel_label_map.get(sample['social-media-channel'], None)
        raw_creator_value=sample['creator']
        raw_logo_label_value=self.logo_label_map.get(sample['logo'], None)
        logo_label_value=torch.tensor([raw_logo_label_value],dtype=torch.float32)


        if isinstance(raw_creator_value, str):
            raw_creator_value = raw_creator_value.strip()  # αφαιρει τυχον κενα απο το string σε αρχη και τελος
            if raw_creator_value.startswith('[') and raw_creator_value.endswith(']'):
                creator_labels = ast.literal_eval(raw_creator_value)  # γινεται πραγματικη python λιστα
            else:
                creator_labels = [raw_creator_value]

        elif isinstance(raw_creator_value, list):
            creator_labels = raw_creator_value
        else:
            creator_labels = []

        assert isinstance(creator_labels, list), "Answer is not a list"

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

        return image, social_media_channel_label_value ,creator_multi_hot_vector, logo_label_value

