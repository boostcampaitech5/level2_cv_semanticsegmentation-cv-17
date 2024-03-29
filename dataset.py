import os
import cv2
import json
import torch
import numpy as np
import albumentations as A
from sklearn.model_selection import GroupKFold

from torch.utils.data import Dataset

from setseed import set_seed

class XRayDataset(Dataset):
    def __init__(self, is_train=True, preprocess = None, augmentation = None, RANDOM_SEED = 21):
        set_seed(RANDOM_SEED)
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.preprocess = preprocess
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.preprocess is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.preprocess(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
            
        if self.augmentation is not None and self.is_train:
            image = np.array(image).astype(np.uint8)
            transform = A.Compose(self.augmentation)
            transformed = transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            image = np.array(image).astype(np.uint64)

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        image = image / 255.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label

def make_dataset(preprocess = None, augmentation = None, RANDOM_SEED = 21):
    train_dataset = XRayDataset(is_train=True, preprocess=preprocess, augmentation = augmentation, RANDOM_SEED = RANDOM_SEED)
    valid_dataset = XRayDataset(is_train=False, preprocess=preprocess, RANDOM_SEED = RANDOM_SEED)

    return train_dataset, valid_dataset

IMAGE_ROOT = "/opt/ml/input/data/train/DCM"
LABEL_ROOT = "/opt/ml/input/data/train/outputs_json"
    
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

pngs = []
for folder in os.listdir(IMAGE_ROOT):
    for file in os.listdir(IMAGE_ROOT +"/"+ folder):
        if file[-3:].lower() == "png":
            file_path = folder + "/" + file
            pngs.append(file_path)
pngs.sort()
    
jsons = []
for folder in os.listdir(LABEL_ROOT):
    for file in os.listdir(LABEL_ROOT +"/"+ folder):
        if file[-4:].lower() == "json":
            file_path = folder + "/" + file
            jsons.append(file_path)
json.sort()
    
jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
    
assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
