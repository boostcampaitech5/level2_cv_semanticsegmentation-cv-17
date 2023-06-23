import os

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models


IMAGE_ROOT = "/opt/ml/input/data/test/DCM"
# IMAGE_ROOT = "/opt/ml/input/data/train/DCM" # psuedo labeling on train data

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


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image, image_name

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            if type(outputs) == type(OrderedDict()):
                outputs = outputs['out']

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            print(image_names)
            for output, image_name in zip(outputs, image_names):
                print(len(output))
                # print(type(output)) # numpy.ndarray
                # print('\vBefore')
                # print(len(output), print(output.shape))
                # for index, out in enumerate(output):
                #     # print(index, np.count_nonzero(output[index]))
                #     if index in [19, 20, 25, 26, 29, 30]:
                #         print(index, np.count_nonzero(output[index]), output[index].shape)
                #         # print(out)
                output = remove_new_classes(output)
                # print('\nAfter')
                # print(len(output), print(output.shape))
                # for index, out in enumerate(output):
                #     # print(index, np.count_nonzero(output[index]))
                #     if index in [19, 20, 25, 26]:
                #         print(index, np.count_nonzero(output[index]), output[index].shape)
                #         # print(out)
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class


def remove_new_classes(output):
    # 'Tratra' : 29 -> 'Trapezium' : 19 / 'Trapezoid' : 20
    # 'Tripis' : 30 -> 'Triquetrum' : 25 / 'Pisiform' : 26

    # print(len(output))
    # print(output)
    # for index, out in enumerate(output):
    #     # print(index, np.count_nonzero(output[index]))
    #     if index in [19, 20, 25, 26, 29, 30]:
    #         print(index, np.count_nonzero(output[index]))
    #         # print(out)
    # print()
    # print('Tratra', np.count_nonzero(output[29]), output[29].shape)
    # print('Tripis', np.count_nonzero(output[30]), output[30].shape)
    Trapezium = np.logical_or(output[19], output[29])
    # print('Trapezium', np.count_nonzero(Trapezium), Trapezium.shape)
    # print(Trapezium)
    output[19] = Trapezium
    Trapezoid = np.logical_or(output[20], output[29])
    # print('Trapezoid', np.count_nonzero(Trapezoid), Trapezoid.shape)
    # print(Trapezoid)
    output[20] = Trapezoid
    Triquetrum = np.logical_or(output[25], output[30])
    # print('Triquetrum', np.count_nonzero(Triquetrum), Triquetrum.shape)
    # print(Triquetrum)
    output[25] = Triquetrum
    Pisiform = np.logical_or(output[26], output[30])
    # print('Pisiform', np.count_nonzero(Pisiform), Pisiform.shape)
    # print(Pisiform)
    output[26] = Pisiform

    output = np.delete(output, 30, 0) # remove 'Tripis'
    output = np.delete(output, 29, 0) # remove 'Tratra'
    # print(len(output))

    return output


def inference(folder_name, preprocess):
    print(f'Start Inference..')
    SAVED_DIR = "/opt/ml/input/code/trained_model/"
    model = torch.load(os.path.join(SAVED_DIR, folder_name + '/best.pt'))
    
    test_dataset = XRayInferenceDataset(transforms=preprocess)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    # test(model, test_loader)
    print(IMAGE_ROOT)
    rles, filename_and_class = test(model, test_loader)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    # df.to_csv(folder_name + "_output.csv", index=False) # "output.csv"
    df.to_csv(os.path.join(SAVED_DIR, folder_name) + '/output.csv', index=False)
    print(f'Inference Done')
    print(f'Result saved in {os.path.join(SAVED_DIR, folder_name)} as output.csv')

if __name__ == "__main__":
    
    import albumentations as A

    folder_name = '[DeepLabV3Plus_resnet34]_[size:(512, 512)]_[loss:BCEWithLogitsLoss()]_[LR:0.0001]_[seed:21]_[epoch:100]_class31'
    inference(folder_name, A.Resize(512, 512))