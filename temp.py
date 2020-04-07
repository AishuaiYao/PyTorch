import numpy as np
import torch
import  cv2



# # hist = np.bincount(
# #     n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
#
n1 = np.array([[1,2,3]
               ,[4,5,6]
               ,[7,8,9]])

# print(n1)

# print(n1.sum(axis=1))




n2 = np.array([[2,2,2]
               ,[2,2,2]
               ,[2,2,2]])

n1,n2 = torch.from_numpy(n1),torch.from_numpy(n2)
n1 = torch.unsqueeze(n1,0)
n2 = torch.unsqueeze(n2,0)
for i,j in zip(n1,n2):
    print(i)
    print(j)
    print('')

#
#
#
# mask = (n1>=3)
# print(mask)
#
#
# n2 = n1[mask]
# print(n2)
#
# n3= 2*n2
# print(n3)
#
#
#
#

#

im = cv2.imread('./FCN/aa.png')
im = np.expand_dims(im,axis=0)
im = torch.from_numpy(im)
label = im.max(dim =3)
label = label[1]
print(1)

#
#
#
#
#












