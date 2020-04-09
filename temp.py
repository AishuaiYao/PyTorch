import numpy as np
import torch
import  cv2



# # hist = np.bincount(
# #     n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
#
n1 = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)

print(n1)

colormap = [[0,0,0],        [128,0,0],      [0,128,0],      [128,128,0],    [0,0,128],
            [128,0,128],    [0,128,128],    [128,128,128],  [64,0,0],       [192,0,0],
            [64,128,0],     [192,128,0],    [64,0,128],     [192,0,128],    [64,128,128],
            [192,128,128],  [0,64,0],       [128,64,0],     [0,192,0],      [128,192,0],
            [0,64,128]]
cm = np.array(colormap)

n2 = cm[n1]
print(n2)













