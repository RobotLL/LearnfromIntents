import numpy as np
import time
import copy
import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from common import ClassifierNet
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


MODELS_DIR = './logs/classifier_models'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

####################################################
#       Merge data
####################################################
roll_train_data = np.load('./train_roll_data.npy')
side_train_data = np.load('./train_side_data.npy')
td_train_data = np.load('./train_td_data.npy')

all_train_data = np.vstack([roll_train_data, side_train_data, td_train_data])

np.save('train_all3_data.npy', all_train_data)

roll_test_data = np.load('./test_roll_data.npy')
side_test_data = np.load('./test_side_data.npy')
td_test_data = np.load('./test_td_data.npy')

all_test_data = np.vstack([roll_test_data, side_test_data, td_test_data])
np.save('test_all3_data.npy', all_test_data)

####################################################
#       Create Train, Valid and Test sets
####################################################
train_data_path = './train_all3_data.npy'
# alert!!!!
test_data_path = './test_all3_data.npy'
bs = 128

train_data = np.load(train_data_path)
test_data = np.load(test_data_path)
dataset_sizes = {'train': len(train_data), 'val': len(test_data)}

####################################################
#             Count train data class weight
####################################################

labels = train_data[:, 14].astype(np.float64).astype(np.int32).copy()

c0 = len(np.where(labels == 0)[0])
c1 = len(np.where(labels == 1)[0])
c2 = len(np.where(labels == 2)[0])
c3 = len(np.where(labels == 3)[0])
c4 = len(np.where(labels == 4)[0])

class_counts = [c0, c1, c2, c3, c4]
num_samples = sum(class_counts)

class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[int(labels[i])] for i in range(int(num_samples))]
train_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)

####################################################
#             Count test data class weight
####################################################
labels = test_data[:, 14].astype(np.float64).astype(np.int32).copy()

c0 = len(np.where(labels == 0)[0])
c1 = len(np.where(labels == 1)[0])
c2 = len(np.where(labels == 2)[0])
c3 = len(np.where(labels == 3)[0])
c4 = len(np.where(labels == 4)[0])

class_counts = [c0, c1, c2, c3, c4]
num_samples = sum(class_counts)

class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[int(labels[i])] for i in range(int(num_samples))]
test_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)

#######################################################
#               Define Dataset Class
#######################################################


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_vector = np.float32(np.array(self.data[idx][:14]).astype(np.float64))
        label = np.int64(np.array(self.data[idx][14]).astype(np.float64))
        img_path = self.data[idx][15]
        image = read_image(img_path)
        image = image.to(torch.float32)
        return input_vector, label, image


#######################################################
#                  Create Dataset
#######################################################

trainset = GraspDataset(train_data)
testset = GraspDataset(test_data)

#######################################################
#                  Define Dataloaders
#######################################################

trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                          shuffle=False, num_workers=0, sampler=train_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                         shuffle=False, num_workers=0, sampler=test_sampler)

#######################################################
#            Define Net, optimizer, etc
#######################################################

net = ClassifierNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# %%
#train_features, train_labels, image = next(iter(trainloader))
# %%
#######################################################
#            Define training process
#######################################################


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            dataloaders = {'train': trainloader, 'val': testloader}
            # Iterate over data.
            for inputs, labels, images in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                images = images.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.5f} Acc: {epoch_acc:.6f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, os.path.join(MODELS_DIR, './best_all3_classifier.pth'))
        print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# %%
#######################################################
#                  Train
#######################################################
if __name__ == '__main__':
    train_model(net, criterion, optimizer, exp_lr_scheduler, num_epochs=200)
