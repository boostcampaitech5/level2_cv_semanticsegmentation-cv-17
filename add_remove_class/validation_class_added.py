import torch
import torch.nn.functional as F
from collections import OrderedDict

from tqdm.auto import tqdm

from metric import dice_coef
from setseed import set_seed

import wandb

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna', 'Tratra', 'Tripis'
] # add new class for multi-label ('Tratra', 'Tripis')

def validation(epoch, model, data_loader, criterion, RANDOM_SEED = 21, thr=0.5):
    set_seed(RANDOM_SEED)
    print(f'Start validation #{epoch:3d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = 31 # 29
        total_loss = 0
        cnt = 0
        
        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            if type(outputs) == type(OrderedDict()):
                outputs = outputs['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            if type(criterion) == list:
                loss = 0
                for losses in criterion:
                    loss += losses[0](outputs, masks) * losses[1]
            else:
                loss = criterion(outputs, masks)

            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            # print(outputs)
            # print(masks)
            
            for mask, image in zip(masks, images):
                # print('\nBefore')
                # print(len(mask), print(mask.shape))
                # print(type(mask))
                # for index, out in enumerate(mask):
                #     # print(index, np.count_nonzero(mask[index]))
                #     if index in [19, 20, 25, 26, 29, 30]:
                #         print(index, torch.count_nonzero(mask[index]), mask[index].shape)
                #         # print(out)
                mask = remove_new_classes(mask)
                # print('\nAfter')
                # print(len(mask), print(mask.shape))
                # for index, out in enumerate(mask):
                #     # print(index, torch.count_nonzero(mask[index]))
                #     if index in [19, 20, 25, 26]:
                #         print(index, torch.count_nonzero(mask[index]), mask[index].shape)
                #         # print(out)
            
            for output, image in zip(outputs, images):
                # print('\nBefore')
                # print(len(output), print(output.shape))
                # print(type(output))
                # for index, out in enumerate(output):
                #     # print(index, np.count_nonzero(output[index]))
                #     if index in [19, 20, 25, 26, 29, 30]:
                #         print(index, torch.count_nonzero(output[index]), output[index].shape)
                #         # print(out)
                output = remove_new_classes(output)
                # print('\nAfter')
                # print(len(output), print(output.shape))
                # for index, out in enumerate(output):
                #     # print(index, torch.count_nonzero(output[index]))
                #     if index in [19, 20, 25, 26]:
                #         print(index, torch.count_nonzero(output[index]), output[index].shape)
                #         # print(out)
            
            dice = dice_coef(outputs, masks, RANDOM_SEED)
            dices.append(dice)

        wandb.log({"sum/valid loss": total_loss / cnt}, step = epoch)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)

    dice_dict = dict()
    for c, d in zip(CLASSES, dices_per_class):
        dice_dict[c] = d
    wandb.log({"class/dice_coef" : dice_dict}, step = epoch)
    
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice


def remove_new_classes(output):
    # 'Tratra' : 29 -> 'Trapezium' : 19 / 'Trapezoid' : 20
    # 'Tripis' : 30 -> 'Triquetrum' : 25 / 'Pisiform' : 26

    # print(len(output))
    # print(output)
    # for index, out in enumerate(output):
    #     # print(index, torch.count_nonzero(output[index]))
    #     if index in [19, 20, 25, 26, 29, 30]:
    #         print(index, torch.count_nonzero(output[index]))
    #         # print(out)
    # print()
    # print('Tratra', torch.count_nonzero(output[29]), output[29].shape)
    # print('Tripis', torch.count_nonzero(output[30]), output[30].shape)
    Trapezium = torch.logical_or(output[19], output[29])
    # print('Trapezium', torch.count_nonzero(Trapezium), Trapezium.shape)
    # print(Trapezium)
    output[19] = Trapezium
    Trapezoid = torch.logical_or(output[20], output[29])
    # print('Trapezoid', torch.count_nonzero(Trapezoid), Trapezoid.shape)
    # print(Trapezoid)
    output[20] = Trapezoid
    Triquetrum = torch.logical_or(output[25], output[30])
    # print('Triquetrum', torch.count_nonzero(Triquetrum), Triquetrum.shape)
    # print(Triquetrum)
    output[25] = Triquetrum
    Pisiform = torch.logical_or(output[26], output[30])
    # print('Pisiform', torch.count_nonzero(Pisiform), Pisiform.shape)
    # print(Pisiform)
    output[26] = Pisiform

    # output = torch.delete(output, 30, 0) # remove 'Tripis'
    # output = torch.delete(output, 29, 0) # remove 'Tratra'
    # print(len(output))

    return output