from torch.autograd import Variable, Function
from torch import nn
import torch

N = 10
x = torch.zeros( size=(3,224,224))
offset = torch.zeros(size=(3,224,224))

offsets_index = Variable(torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]),
                         requires_grad=False).type_as(x).long()
offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
offset = torch.gather(offset, dim=1, index=offsets_index)














