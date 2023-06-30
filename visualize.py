#%%
import numpy as np
import matplotlib.pyplot as plt

from random import *

# import numpy as np #numpy library
# np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity 

# define colors
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 255, 41), (102, 255, 51)
] # add color for new classes ('Tratra', 'Tripis')

# utility function
# this does not care overlap
def label2rgb(label):
    image_size = label.shape[1:] + (3, ) # (512, 512) -> (512, 512, 3)
    image = np.zeros(image_size, dtype=np.uint8) # '0'으로 구성된 (512, 512, 3) shape의 matrix 준비
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i] # 각 class에 대한 annotation을 (512, 512, 3) matrix에 RGB값을 할당
        
    return image

def visualize_dataset(dataset):
    print(dataset)
    image, label = dataset[randint(0, 500)] # train_dataset[0]
    print(image.shape, label.shape)
    print(len(dataset))
    # print(image)
    # print(label)
    
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image[0])    # remove channel dimension
    ax[1].imshow(label2rgb(label))

    plt.show()


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from setseed import set_seed
    # from dataset import make_dataset
    from resized_dataset import make_resized_dataset
    # from resized_class_added_dataset import make_resized_class_added_dataset
    from resized_class_removed_dataset import make_resized_class_removed_dataset
    
    BATCH_SIZE = 8
    resize = 512
    RANDOM_SEED = 21

    set_seed(RANDOM_SEED)

    # ensemble_ver2
    train_dataset2, valid_dataset2 = make_resized_dataset(RANDOM_SEED = RANDOM_SEED)
    # ensemble_ver1
    train_dataset1, valid_dataset1 = make_resized_class_removed_dataset(RANDOM_SEED = RANDOM_SEED)

    visualize_dataset(train_dataset2)
    visualize_dataset(train_dataset1)
# %%
