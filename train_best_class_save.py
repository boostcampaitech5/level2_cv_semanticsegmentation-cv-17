from validation import validation
from setseed import set_seed

import os
import torch
import numpy as np
import random
from collections import OrderedDict
import datetime
import wandb

SAVED_DIR = "/opt/ml/input/code/trained_model/"
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

if not os.path.isdir(SAVED_DIR):                                                           
    os.mkdir(SAVED_DIR)

def train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS = 30, VAL_EVERY = 1, folder_name = 'last_model', RANDOM_SEED = 21,PATIENCE=5):
    print(f'Start training..')
    set_seed(RANDOM_SEED)

    if not os.path.isdir(os.path.join(SAVED_DIR, folder_name)):                                                           
        os.mkdir(os.path.join(SAVED_DIR, folder_name))
    else:
        idx = 2
        print()
        while os.path.isdir(os.path.join(SAVED_DIR, folder_name) + '_' + str(idx)):
            idx += 1 
        folder_name = folder_name + '_' + str(idx)
        os.mkdir(os.path.join(SAVED_DIR, folder_name))
    print("result model will be saved in {}".format(os.path.join(SAVED_DIR, folder_name)))

    wandb.init(
        # set the wandb project where this run will be logged
        entity = 'sixseg_semantic_seg',
        project="boostcamp_level2_semantic_segmentation",
        name = folder_name,
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": optimizer.param_groups[0]['lr'],
        "architecture": model.name,
        "loss": criterion,
        "epochs": NUM_EPOCHS,
        "seed": RANDOM_SEED
        }
    )
    
    n_class = 29
    best_dice = 0.
    best_dice_epoch = 0
    best_dices_per_class = [0. for _ in range(29)]
    best_dices_per_class_epoch = [0 for _ in range(29)]
    loss = 0
    early_stop=0

    for epoch in range(NUM_EPOCHS):
        print(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
            f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
            f'Step [0/{len(train_loader)}]'
        )
        
        model.train()

        for step, (images, masks) in enumerate(train_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            # inference
            outputs = model(images)
            if type(outputs) == type(OrderedDict()):
                outputs = outputs['out']
            
            # loss 계산
            if type(criterion) == list:
                loss = 0
                for losses in criterion:
                    loss += losses[0](outputs, masks) * losses[1]
            else:
                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice, dices_per_class = validation(epoch + 1, model, val_loader, criterion, RANDOM_SEED, 0.5)

            for i in range(29):
                if best_dices_per_class[i] < dices_per_class[i].item():
                    print(f"{CLASSES[i]:<12}: {dices_per_class[i].item():.4f}")
                    best_dices_per_class[i] = dices_per_class[i].item()
                    save_best_class_model(model, folder_name, i)
                    best_dices_per_class_epoch[i] = epoch + 1
                else:
                    print(f"{CLASSES[i]:<12}: {dices_per_class[i].item():.4f}")
            
            if best_dice < dice:
                print(f"New best average dice coefficient at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {os.path.join(SAVED_DIR, folder_name)}")
                best_dice = dice
                best_dice_epoch = epoch + 1
                early_stop = 0
                save_best_model(model, folder_name)
            else:
                print(f"Performance at epoch: {epoch + 1}, {dice:.4f}")
                print(f"Best performance was at epoch: {best_dice_epoch}, {best_dice:.4f}")
                early_stop += 1
                if early_stop >= PATIENCE and best_dice > 0.5:
                    print("No more update")
                    break
        wandb.log({"sum/train_loss": loss, "sum/dice_coef": dice, "parameter/lr" : optimizer.param_groups[0]['lr']}, step = epoch + 1)

    for i in range(29):
        print(f"{CLASSES[i]:<12}: {best_dices_per_class[i]:.4f} at epoch {best_dices_per_class_epoch[i]}")

    wandb.finish()
    
    return folder_name

def save_best_model(model, folder_name):
    output_path = os.path.join(SAVED_DIR, folder_name)
    if not os.path.isdir(output_path):                                                           
        os.mkdir(output_path)
    torch.save(model, output_path + '/best.pt')

def save_best_class_model(model, folder_name, idx):
    output_path = os.path.join(SAVED_DIR, folder_name) + '/best_class'
    if not os.path.isdir(output_path):                                                           
        os.mkdir(output_path)
    torch.save(model, output_path + '/best_' + CLASSES[idx] + '.pt')