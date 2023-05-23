from __future__ import print_function
from __future__ import division

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from torchvision.utils import save_image

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classes', type=int, choices=[2, 37], default=2, nargs='?')
parser.add_argument('-e', '--epochs', type=int, default=15, nargs='?')
parser.add_argument('-n', '--numlayers', type=int, choices=range(1, 6), default=1, nargs='?')
parser.add_argument('-b', '--update_batch_norm_params', action='store_true', default=False)
parser.add_argument('-d', '--data_augmentation', action='store_true', default=False)
parser.add_argument('-t', '--training_data', choices=['small', 'medium', 'large'], default='large')
parser.add_argument("--sophisticated_data_augs", choices=['none', 'cutmix', 'mixup', 'erase'], default=False)
parser.add_argument("--only_update_bn_params", action="store_true", default=False)
args = parser.parse_args()

if args.only_update_bn_params and not args.update_batch_norm_params:
    print("Update or not update, choose one!")
    exit()

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = f"./oxford-iiit-pet/dataset-" + args.training_data + "-train/" + str(args.classes) + "-class"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = args.classes

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# Number of epochs to train for 
num_epochs = args.epochs


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, args, p_mixup=0.5):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                ############################
                # Apply Mixup augmentation #
                ############################
                if phase == "train":
                    if args.sophisticated_data_augs == "mixup":
                        p = np.random.rand()
                        if p < p_mixup:
                            samples, labels = mixup(inputs, labels, 0.5)
                    elif args.sophisticated_data_augs == "cutmix":
                        p = np.random.rand()
                        if p < p_mixup:
                            samples, labels = cutmix(inputs, labels, 0.5)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)

                    if args.sophisticated_data_augs == "mixup" and phase == "train" and p < p_mixup:
                        loss = mixup_criterion(outputs, labels)
                        labels = labels[1]
                    elif args.sophisticated_data_augs == "cutmix" and phase == "train" and p < p_mixup:
                        loss = cutmix_criterion(outputs, labels)
                        labels = labels[1]
                    else:
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'train':
                epoch_acc = 0
            else:
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, num_layers_trained, args):
    for param in model.parameters():
        param.requires_grad = False

    target_layers = ["layer" + str(i) for i in range(4, 5 - num_layers_trained, -1)]
    for name, param in model.named_parameters():
        for layer in target_layers:
            if layer in name:
                if (not args.update_batch_norm_params) and 'bn' in name:
                    continue
                elif args.only_update_bn_params and 'bn' not in name:
                    continue
                else:
                    param.requires_grad = True


def initialize_model(model_name, num_classes, numlayers, args):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet34
        """
        model_ft = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, numlayers, args)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_test_accuracy(model, dataloaders):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    phase = "test"
    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # Do not track history
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


# Copied and edited from https://www.kaggle.com/code/riadalmadani/fastai-effb0-base-model-birdclef2023
def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


# Copied and edited from https://www.kaggle.com/code/riadalmadani/fastai-effb0-base-model-birdclef2023
def mixup_criterion(preds, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


# Copied and edited from https://www.kaggle.com/code/riadalmadani/fastai-effb0-base-model-birdclef2023
def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets


def cutmix_criterion(preds, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, args.numlayers, args)

# Print the model we just instantiated
# print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation

if args.data_augmentation:
    train_transforms = [
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
else:
    train_transforms = [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

if args.sophisticated_data_augs == 'erase':
    train_transforms.insert(-1, transforms.RandomErasing())

data_transforms = {
    'train': transforms.Compose(train_transforms),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train', 'val', 'test']}
# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in
    ['train', 'val', 'test']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.

# Set up the loss fxn
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
layer_max_lrs = [1e-3, 1e-3, 1e-4, 1e-5, 1e-4]
params_dictionaries = []
for i in range(args.numlayers):
    params_dictionaries.append({"params": []})

for name2, param2 in model_ft.named_parameters():
    if "fc" in name2:
        params_dictionaries[0]["params"].append(param2)

target_layers = ["layer" + str(i) for i in range(4, 5 - args.numlayers, -1)]
for i in range(len(target_layers)):
    for name, param in model_ft.named_parameters():
        if target_layers[i] in name:
            # Skip batch norm layers if we are not updating them
            if (not args.update_batch_norm_params) and 'bn' in name:
                continue
            elif args.only_update_bn_params and 'bn' not in name:
                continue
            params_dictionaries[i + 1]["params"].append(param)

optimizer_ft = optim.Adam(params_dictionaries, weight_decay=1e-5, lr=1e-7)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer_ft, max_lr=layer_max_lrs[:args.numlayers], pct_start=0.8,
                                          steps_per_epoch=len(dataloaders_dict["train"]), epochs=num_epochs)

for name, param in model_ft.named_parameters():
    if param.requires_grad:
        print("\t", name)

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, num_epochs, args)
get_test_accuracy(model_ft, dataloaders_dict)
