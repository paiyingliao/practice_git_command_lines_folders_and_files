# Created on Feb. 4th by Pai-Ying Liao (liao119, PUID: 0029934248) for ECE 695 DL HW2
import argparse
from torch.utils.data import DataLoader, Dataset
import glob
import torchvision.transforms as tvt
from PIL import Image
from numpy import asarray
import torch

torch.manual_seed(1)

parser = argparse.ArgumentParser(description='HW02 Task2')
parser.add_argument('--imagenet_root', type=str, required=True)
parser.add_argument('--class_list', nargs='*', type=str, required=True)
args, args_other = parser.parse_known_args()

class AnimalDataset(Dataset):
    def __init__(self, transform, type):
        self.imgs = []
        for animal_kind in args.class_list:
            for img in glob.glob(args.imagenet_root + "/" + type + "/" + animal_kind + "/*"):
                self.imgs.append(img)
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        im = Image.open(self.imgs[idx])
        im_array = asarray(im)
        im_normalized = im_array / 255.0
        im_transformed = self.transform(im_normalized)
        label = torch.zeros(2)
        s = self.imgs[idx]
        while s[-1] != '/':
            s = s[:-1]
        s = s[:-1]
        if s[-3:] == 'cat':
            label[0] = 1
        else:
            label[1] = 1
        ret = {'image': im_transformed, 'label': label}
        return ret

transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = AnimalDataset(transform, 'Train')
train_data_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
val_dataset = AnimalDataset(transform, 'Val')
val_data_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=True)

dtype = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 40
D_in, H1, H2, D_out = 3 * 64 * 64, 1000, 256, 2
w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
w2 = torch.randn(H1, H2, device=device, dtype=dtype)
w3 = torch.randn(H2, D_out, device=device, dtype=dtype)
learning_rate = 1e-9

loss_list_train = []

for t in range(epochs):

    for i, data in enumerate(train_data_loader):
        x, y = data['image'], data['label']
        x = x.to(device)
        y = y.to(device)
        x = x.view(x.size(0), -1)

        # forward
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3)

        # loss calculation
        loss = (y_pred - y).pow(2).sum().item()
        y_error = y_pred - y
        if len(loss_list_train) == t:
            loss_list_train.append(loss)
        else:
            loss_list_train[-1] += loss

        # backward propagation
        grad_w3 = h2_relu.transpose(0, 1).mm(2 * y_error)
        h2_error = 2.0 * y_error.mm(w3.transpose(0, 1))
        h2_error[h2_error < 0] = 0
        grad_w2 = h1_relu.transpose(0, 1).mm(2 * h2_error)
        h1_error = 2.0 * h2_error.mm(w2.transpose(0, 1))
        h1_error[h1_error < 0] = 0
        grad_w1 = x.transpose(0, 1).mm(2 * h1_error)

        # weight update
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3

    print('Epoch %d:\t %0.4f'%(t, loss_list_train[-1]))

torch.save({'w1': w1, 'w2': w2, 'w3': w3}, './wts.pkl')

correct, total, val_loss = 0, 0, 0

for i, data in enumerate(val_data_loader):
    x, y = data['image'], data['label']
    x = x.to(device)
    y = y.to(device)
    x = x.view(x.size(0), -1)

    # forward
    h1 = x.mm(w1)
    h1_relu = h1.clamp(min=0)
    h2 = h1_relu.mm(w2)
    h2_relu = h2.clamp(min=0)
    y_pred = h2_relu.mm(w3)

    # loss calculation
    loss = (y_pred - y).pow(2).sum().item()
    y_error = y_pred - y
    val_loss += loss

    predicted = torch.max(y_pred.data, 1)[1]
    answer = torch.max(y.data, 1)[1]
    total += len(predicted)
    correct += int((predicted == answer).sum())

print("\nVal Loss:\t{}".format(val_loss))
print("Val Accuracy = {}%".format(correct * 100.0 / total))