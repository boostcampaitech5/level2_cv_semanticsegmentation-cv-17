import os
import cv2
import json
import torch
import numpy as np

from sklearn.model_selection import GroupKFold

from torch.utils.data import Dataset

from setseed import set_seed

from add_class import get_new_class_mask

IMAGE_ROOT = "/opt/ml/input/data/train/resized_DCM"
LABEL_ROOT = "/opt/ml/input/data/train/resized_outputs_json" # "/opt/ml/input/data/train/resized_class_added_outputs_json"
    
CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna', 'Tratra', 'Tripis'
] # add new class for multi-label ('Tratra', 'Tripis')

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
}
    
jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
}
    
jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
    
assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
    
pngs = sorted(pngs)
jsons = sorted(jsons)


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
        self.augmentation = augmentation # augmentation
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        # image = image / 255.
        
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
        temp = {}
        nonzero = {}
        for index, ann in enumerate(annotations):
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            # print(c, class_ind)
            # print('points ', points)

            # polygon to mask
            # if len(points) != 0:
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)

            # get new class
            # 'Trapezium' : 19 / 'Trapezoid' : 20 -> 'Tratra' : 29 (new)
            # 'Triquetrum' : 25 / 'Pisiform' : 26 -> 'Tripis' : 30 (new)
            if c in ['Trapezium', 'Trapezoid', 'Triquetrum', 'Pisiform']:
                temp[c] = class_label
                nonzero[c] = np.count_nonzero(class_label)
            else:
                label[..., class_ind] = class_label
            
            if index == len(annotations) - 1:
                # print('before')
                # print(temp)
                # print(nonzero)
                temp, nonzero = get_new_class_mask(temp, nonzero)
                # print('after')
                # print(temp)
                # print(nonzero)
                
                for key, value in temp.items():
                    class_ind = CLASS2IND[key]
                    label[..., class_ind] = value
                # print(label)
            
        
        if self.preprocess is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.preprocess(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
        

        if self.augmentation is not None: # and self.is_train:
            # image = image.astype(np.uint8) # brightness
            aug = A.Compose(self.augmentation)
            aug_result = aug(image=image, mask=label)
            image = aug_result['image']
            label = aug_result['mask']
            # image = image.astype(np.uint64)
        
        image = image / 255.

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label

def make_resized_class_added_dataset(preprocess = None, augmentation = None, RANDOM_SEED = 21):
    train_dataset = XRayDataset(is_train=True, preprocess=preprocess, augmentation = augmentation, RANDOM_SEED = RANDOM_SEED)
    valid_dataset = XRayDataset(is_train=False, preprocess=preprocess, RANDOM_SEED = RANDOM_SEED)

    return train_dataset, valid_dataset




# save_resized_image & json
# play in dataset.py

from PIL import Image
from tqdm.auto import tqdm
import albumentations as A

IMAGE_RESIZED_ROOT = "/opt/ml/input/data/train/resized_DCM"
def save_resized_image():
    # save resized image (2048, 2048) -> (512, 512)
    tf = A.Resize(512, 512)

    if not os.path.isdir(IMAGE_RESIZED_ROOT):
        os.mkdir(IMAGE_RESIZED_ROOT)

    for image_name in tqdm(pngs):
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = cv2.imread(image_path)
        image = tf(image = image)

        if not os.path.isdir(os.path.join(IMAGE_RESIZED_ROOT, image_name[:5])):
            os.mkdir(os.path.join(IMAGE_RESIZED_ROOT, image_name[:5]))

        image_path = os.path.join(IMAGE_RESIZED_ROOT, image_name)

        image = Image.fromarray(image['image'])
        image.save(image_path)
        

LABEL_RESIZED_ROOT = "/opt/ml/input/data/train/resized_outputs_json"
def save_resized_json():
    # save resized json (2048, 2048) -> (512, 512)
    resize = (512,512)
    size = (2048, 2048)
    origin_root = "../data/train/outputs_json"
    resize_root = "../data/train/resized_outputs_json"

    path = []
    for folder in os.listdir(origin_root):
        for file in os.listdir(origin_root +"/"+ folder):
            if file[-4:].lower() == "json":
                file_path = folder + "/" + file
                path.append(file_path)

    for file in path:
        ORIGIN_ROOT = origin_root + "/" + file
        RESIZE_ROOT = resize_root + "/" + file

        folder_name = file.split("/")[0]    
        os.makedirs(resize_root + "/" + folder_name, exist_ok=True)

        with open(ORIGIN_ROOT, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for anno in data["annotations"]:
            resize_point = []
            for point in anno["points"]:
                pnt = [point[0]*resize[0]//size[0],point[1]*resize[1]//size[1]]
                if pnt not in resize_point:
                    resize_point.append(pnt)
            anno["points"] = resize_point
        
        with open( RESIZE_ROOT,'w',  encoding='utf-8') as train_writer:
                json.dump(data, train_writer, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    save_resized_json()