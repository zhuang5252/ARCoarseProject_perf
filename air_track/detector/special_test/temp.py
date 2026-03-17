import cv2
import torch

path = '/home/csz_changsha/lcf/data/rgb/dlc_T18_120180_4807.20s.jpg'

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (768, 576))
tensor = torch.from_numpy(img)
print(tensor.shape)

tensor = tensor.reshape(768, 576)

out = tensor.numpy()
cv2.imwrite('test.jpg', out)
