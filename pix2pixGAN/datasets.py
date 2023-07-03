import glob
import random
import os
import cv2
import json
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def find_file(ROOT, extension):
    FILES = []
    LEN = len(extension)
    for folder in os.listdir(ROOT):
        for file in os.listdir(ROOT +"/"+ folder):
            if file[-LEN:].lower() == extension:
                file_path = ROOT +"/"+ folder + "/" + file
                FILES.append(file_path)
    FILES.sort()
    return FILES


def json_to_img(file, img_size, num_class):
    
        label_path = file

        # process a label of shape (H, W, NC)
        label_shape = (num_class, ) + img_size
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r", encoding="UTF-8") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(img_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[class_ind, ...] = class_label

        return label


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)


        if mode == "train" or mode == "test":
            file = ["train"]
            
            FI = find_file(root[0], "png")
            file.append(FI)
            
            self.files = file
            
        if mode == "val":
            file = ["val"]
            
            FI = find_file(root[0], "png")
            file.append(FI)

            FI = find_file(root[1], "json")
            file.append(FI)
            
            self.files = file
            

    def __getitem__(self, index):
        
        idx = index % len(self.files[1])
        
        if self.files[0] == "train":
            img = cv2.imread(self.files[1][idx])
            
        elif self.files[0] == "val":
            img_size = (512,512)
            
            img_1 = cv2.imread(self.files[1][idx])
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            img_2 = json_to_img(self.files[2][idx], img_size, num_class=29)
            
            img_sum = img_2.sum()
            img_2 = np.zeros(img_size , dtype=np.uint8)
            img_3 = np.zeros(img_size , dtype=np.uint8)
            for i,cls in enumerate(img_2):
                if i%2:
                    img_2 += img_1*cls*(1 - (img_sum-cls)//2 )
                else:
                    img_3 += img_1*cls*(1 - (img_sum-cls)//2 )
            
            img_1 = np.expand_dims(img_1, axis=2)
            img_2 = np.expand_dims(img_2, axis=2)
            img_3 = np.expand_dims(img_3, axis=2)
            
            img = np.concatenate((img_1, img_2, img_3), axis=2)
            
        
        #img = Image.open(self.files[index % len(self.files)])
        
        h,w,c = img.shape
        
        img_A = img[:,:w//2,:]
        img_B = img[:,w//2:,:]

        if np.random.random() < 0.5:
            img_A = Image.fromarray(img_A[:, ::-1, :], "RGB")
            img_B = Image.fromarray(img_B[:, ::-1, :], "RGB")
            
        else:
            img_A = Image.fromarray(img_A, "RGB")
            img_B = Image.fromarray(img_B, "RGB")
            

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
    
    
    
CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}