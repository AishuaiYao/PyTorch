import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable, Function


class DeformConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=1, bias=True):
        super(DeformConv2d, self).__init__()
        self.ks = kernel_size
        self.pad = padding
        self.zero_pad = nn.ZeroPad2d(padding)
        self.conv2d = nn.Conv2d(input_channel, output_channel, kernel_size, 3, bias)

    def forward(self, x, offset):
        b, n, w, h = offset.size()
        direction = n // 2

        """把offset的xyxyxy转为xxxyyy"""
        x_index = torch.arange(0, n, 2)
        y_index = torch.arange(1, n + 1, 2)
        offset_index = Variable(torch.cat([x_index, y_index]), requires_grad=False)
        offset_index = offset_index.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand(offset.size()).cuda()
        offset = torch.gather(offset, dim=1, index=offset_index)

        # 定义感受场大小，论文里的R域
        """
        -1 0 1      -1 -1 -1 
        -1 0 1       0  0  0
        -1 0 1       1  1  1
        比如一个在特征图F上的一个点p, 从上面两个matrix的变量对应值，对应着(-1, -1),(0, -1)...。
        意味着p的左上和上。是一个位置指示
        """
        _ = range(-(self.ks - 1) // 2, (self.ks - 1) // 2 + 1)
        r_x, r_y = np.meshgrid(_, _)
        r_flatten = np.concatenate([r_x.flatten(), r_y.flatten()])
        # [x:-1  0  1 -1  0  1 -1  0  1 y:-1 -1 -1  0  0  0  1  1  1]
        r_flatten = np.reshape(r_flatten, (1, n, 1, 1))  # [1, 18, 1, 1]
        R = Variable(torch.from_numpy(r_flatten), requires_grad=False)

        _ = range(0, h)
        x_pos, y_pos = np.meshgrid(_, _)
        x_pos = x_pos.flatten().reshape(1, 1, h, w).repeat(direction, axis=1)
        y_pos = y_pos.flatten().reshape(1, 1, h, w).repeat(direction, axis=1)
        original = np.concatenate((x_pos, y_pos), axis=1)

        """
        position是c个通道每个位置的原始坐标
        """
        original = Variable(torch.from_numpy(original), requires_grad=False)
        position = (original + R).type_as(offset)
        """
        这里p是特征图每个位置的坐标,offset是网络预测出的每个位置的偏移量,所以这里相加就是最终的虚拟位置
        
        """
        new_position = position + offset
        new_position = new_position.contiguous().permute(0, 2, 3, 1)

        """
        下面是开始对超出特征图尺寸的值进行约束,同时把左上右下左下右上的位置坐标也都提出了
        """
        lt = Variable(new_position.data, requires_grad=False).floor()
        lt = torch.floor(lt)
        rb = lt + 1

        u = torch.clamp(lt[..., :direction], 0, w-1)
        v = torch.clamp(lt[..., direction:], 0, h-1)
        lt = torch.cat([u, v], dim=-1)

        u = torch.clamp(rb[..., :direction], 0, w-1)
        v = torch.clamp(rb[..., direction:], 0, h-1)
        rb = torch.cat([u, v], dim=-1)

        lb = torch.cat((lt[..., :direction], rb[..., direction:]), dim=-1)
        rt = torch.cat((rb[..., :direction], lt[..., direction:]), dim=-1)

        g_lt = self.g(lt[..., :direction], new_position[..., :direction]) * self.g(lt[..., direction:], new_position[..., direction:])
        g_rb = self.g(rb[..., :direction], new_position[..., :direction]) * self.g(rb[..., direction:], new_position[..., direction:])
        g_lb = self.g(lb[..., :direction], new_position[..., :direction]) * self.g(lb[..., direction:], new_position[..., direction:])
        g_rt = self.g(rt[..., :direction], new_position[..., :direction]) * self.g(rt[..., direction:], new_position[..., direction:])

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, lt, direction)
        x_q_rb = self._get_x_q(x, rb, direction)
        x_q_lb = self._get_x_q(x, lb, direction)
        x_q_rt = self._get_x_q(x, rt, direction)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.ks)
        out = self.conv2d(x_offset)

        return out

    def g(self, p, q):
        return torch.max(torch.zeros_like(p), 1 - torch.abs(p - q))


    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1).long()

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


