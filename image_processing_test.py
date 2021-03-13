from PIL import Image
from numpy import asarray
import torchvision.transforms as tvt
import torch

im = Image.open('/Users/liao/Desktop/HW/liao119_HW2/Train/cat/471345_f090f25d64.jpg')

im_array = asarray(im)
print(type(im_array))
print(im_array.shape)
print(im_array[0][0])

im_array = im_array / 255.0
print(im_array[0][0])

# im_tensor = tvt.ToTensor()(im_array)
# print(type(im_tensor))
# print(im_tensor.shape)
# print(im_tensor[0])

transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
im_array = transform(im_array)
print(type(im_array))
print(im_array.shape)
print(im_array[0][0])

z = torch.zeros(2)
z[0] = 1
print(z)
print(type(z))
print(z[0].dtype)