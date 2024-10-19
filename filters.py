import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def get_filter_by_name(filter_name, device, filter_params):
    if filter_name == 'gaussian':
        filter = GaussianFilter(**filter_params).to(device)
    elif filter_name == 'bilateral':
        filter = BilateralFilter_OutofMemory(**filter_params).to(device)
    elif filter_name == 'movavg':
        filter = MovAvgFilter(**filter_params).to(device)
    elif filter_name == 'sepgauss':
        filter = SeparateGaussianFilter(**filter_params).to(device)
    elif filter_name == 'sepvertgauss':
        filter = SeparateVerticalGaussianFilter(**filter_params).to(device)
    elif filter_name == 'sepvertbilat':
        filter = SeparateVerticalBilateralFilter(**filter_params).to(device)
    elif filter_name == 'guided':
        filter = GuidedFilter(**filter_params).to(device)
    else:
        filter = None
    return filter


# https://github.com/proceduralia/pytorch-neural-enhance/blob/master/loss.py
def gaussian_kernel(kernel_size=[5, 5, 5], sigma_size=None, channels=3):
    # initialize guassian kernel
    if sigma_size is None:
        sigma_size = [
            0.3 * ((ksize - 1) / 2.0 - 1) + 0.8 for ksize in kernel_size
        ]
    kernel = 1
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    meshgrids = torch.meshgrid([torch.arange(ksize) for ksize in kernel_size],
                               indexing='ij')
    for size, std, mgrid in zip(kernel_size, sigma_size, meshgrids):
        mean = (size - 1) / 2
        kernel *= torch.exp(-(((mgrid - mean)**2) / ((std)**2) / 2))

    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())  # [3,1,x,y,z]
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    return kernel


def rbf_kernel(kernel_size=[5, 5, 5],
               channels=3,
               name="gaussian",
               sigma_size=None):
    # initialize guassian kernel

    if name == "edw":
        kernel = 0
    elif name == "gaussian" or name == 'inv_quad':
        kernel = 1

    if sigma_size is None:
        sigma_size = [
            0.3 * ((ksize - 1) / 2.0 - 1) + 0.8 for ksize in kernel_size
        ]
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    meshgrids = torch.meshgrid([torch.arange(ksize) for ksize in kernel_size],
                               indexing='ij')
    for size, mgrid, std in zip(kernel_size, meshgrids, sigma_size):
        mean = (size - 1) / 2
        dis = (mgrid - mean)**2
        if name == "edw":
            kernel += dis
        elif name == "gaussian":
            kernel *= torch.exp(-(dis / (std**2)))
        elif name == "inv_quad":
            kernel *= 1 / torch.sqrt(1 + dis / (std**2))

    if name == "edw":
        kernel = torch.exp(-torch.sqrt(kernel))

    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())  # [3,1,x,y,z]
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    return kernel


# https://github.com/sunny2109/bilateral_filter_Pytorch/blob/main/bilateral_filter.py
class GaussianFilter(nn.Module):

    def __init__(self,
                 kernel_size,
                 sigma_space=None,
                 channels=None,
                 device='cpu',
                 fp16=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = len(kernel_size)
        self.pad_size = tuple([(ksize - 1) // 2 for ksize in kernel_size])
        self.channels = channels
        if channels is None:
            self.channels = self.dim
        self.weight = gaussian_kernel(kernel_size,
                                      sigma_size=sigma_space,
                                      channels=self.channels).to(device)
        if fp16:
            self.weight = self.weight.half()
        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def forward(self, x):
        x_filtered = self.conv(
            x,
            weight=self.weight,
            stride=1,
            groups=self.channels,
            padding=self.pad_size,
        )
        return x_filtered


class SeparateGaussianFilter(nn.Module):

    def __init__(self, kernel_size, sigma_space=None, device='cpu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = len(kernel_size)
        self.weights = [
            gaussian_kernel(kernel_size=[
                1,
            ] * idx + [
                kernel_size[idx],
            ] + [
                1,
            ] * (self.dim - idx - 1),
                            sigma_size=sigma_space,
                            channels=1).to(device) for idx in range(self.dim)
        ]
        # for idx in range(self.dim):
        #     print(idx, self.weights[idx].shape)

        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def forward(self, x):
        # [b,3/2,x,y,z]
        x_filtered_list = []
        for idx in range(self.dim):
            x_filtered = self.conv(
                x[:, idx].unsqueeze(1),
                weight=self.weights[idx],
                stride=1,
                padding='same',
            )
            x_filtered_list.append(x_filtered)
        return torch.cat(x_filtered_list, dim=1)


class SeparateVerticalGaussianFilter(nn.Module):

    def __init__(self, kernel_size, sigma_space=None, device='cpu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = len(kernel_size)
        if sigma_space is None:
            sigma_space = [
                0.3 * ((ksize - 1) / 2.0 - 1) + 0.8 for ksize in kernel_size
            ]

        if self.dim == 3:
            self.weights = [
                gaussian_kernel(
                    kernel_size=[1, kernel_size[1], kernel_size[2]],
                    sigma_size=[1, sigma_space[1], sigma_space[2]],
                    channels=1).to(device),
                gaussian_kernel(
                    kernel_size=[kernel_size[0], 1, kernel_size[2]],
                    sigma_size=[sigma_space[0], 1, sigma_space[2]],
                    channels=1).to(device),
                gaussian_kernel(
                    kernel_size=[kernel_size[0], kernel_size[1], 1],
                    sigma_size=[sigma_space[0], sigma_space[1], 1],
                    channels=1).to(device)
            ]
        # for idx in range(self.dim):
        #     print(idx, self.weights[idx].shape)

        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def forward(self, x):
        # [b,3/2,x,y,z]
        x_filtered_list = []
        for idx in range(self.dim):
            x_filtered = self.conv(
                x[:, idx].unsqueeze(1),
                weight=self.weights[idx],
                stride=1,
                padding='same',
            )
            x_filtered_list.append(x_filtered)
        return torch.cat(x_filtered_list, dim=1)


class SeparateVerticalBilateralFilter(nn.Module):

    def __init__(self,
                 kernel_size,
                 sigma_space=None,
                 sigma_density=1.0,
                 device='cpu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = len(kernel_size)
        if sigma_space is None:
            sigma_space = [
                0.3 * ((ksize - 1) / 2.0 - 1) + 0.8 for ksize in kernel_size
            ]
        self.sigma_density = sigma_density

        if self.dim == 3:
            self.weights = [
                gaussian_kernel(
                    kernel_size=[1, kernel_size[1], kernel_size[2]],
                    sigma_size=[1, sigma_space[1], sigma_space[2]],
                    channels=1).to(device),
                gaussian_kernel(
                    kernel_size=[kernel_size[0], 1, kernel_size[2]],
                    sigma_size=[sigma_space[0], 1, sigma_space[2]],
                    channels=1).to(device),
                gaussian_kernel(
                    kernel_size=[kernel_size[0], kernel_size[1], 1],
                    sigma_size=[sigma_space[0], sigma_space[1], 1],
                    channels=1).to(device)
            ]
        self.pad_size = tuple([(ksize - 1) // 2 for ksize in self.kernel_size])
        # for idx in range(self.dim):
        #     print(idx, self.weights[idx].shape)

        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def calculate(self, x, cur_dim):
        # 对输入的x，计算与dim垂直的平面
        # 接下来以dim=0为例，计算结果的维度，x:[b,1,x,y,z]
        pad_size = list(self.pad_size)
        pad_size[cur_dim] = 0
        kernel_size = list(self.kernel_size)
        kernel_size[cur_dim] = 1

        x_pad = F.pad(
            x,
            pad=[xpad for xpad in pad_size[::-1] for _ in range(2)],
            mode="replicate",
        )

        x_patches = x_pad.unfold(2, kernel_size[0],
                                 1).unfold(3, kernel_size[1], 1)
        x_unsqz = x.unsqueeze(-1).unsqueeze(-1)
        if self.dim == 3:
            x_patches = x_patches.unfold(4, kernel_size[2], 1)
            x_unsqz = x_unsqz.unsqueeze(-1)

        # unfold表示滑动窗口的操作。
        diff_density = x_patches - x_unsqz
        weight_density = torch.exp(-(diff_density**2) /
                                   (2 * self.sigma_density**2))

        sum_dim = tuple(range(-1, -self.dim - 1, -1))
        weight_density = weight_density / weight_density.sum(dim=sum_dim,
                                                             keepdim=True)

        # Keep same shape with weight_density
        # (1,1)中的第二个是由于x的channels是1.
        weight_space_dim = (1, 1) + self.dim * (1, ) + tuple(kernel_size)
        weight_space = self.weights[cur_dim].view(*weight_space_dim)

        # get the final kernel weight
        weight = weight_density * weight_space
        weight_sum = weight.sum(dim=sum_dim)
        x = (weight * x_patches).sum(dim=sum_dim) / weight_sum

        return x

    def forward(self, x):
        # [b,3/2,x,y,z]
        x_filtered_list = []
        for idx in range(self.dim):
            x_filtered = self.calculate(
                x[:, idx].unsqueeze(1),
                cur_dim=idx,
            )
            x_filtered_list.append(x_filtered)
        return torch.cat(x_filtered_list, dim=1)


# moving average
class MovAvgFilter(nn.Module):

    def __init__(self, kernel_size, device='cpu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = len(kernel_size)
        self.pad_size = tuple([(ksize - 1) // 2 for ksize in kernel_size])
        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d
        self.weights = [
            torch.ones([[
                1,
                1,
            ] + [
                1,
            ] * idx + [
                kernel_size[idx],
            ] + [
                1,
            ] * (self.dim - idx - 1)]).to(device) / kernel_size[idx]
            for idx in range(self.dim)
        ]

    def forward(self, x):
        # [b,3/2,x,y,z]
        x_filtered_list = []
        for idx in range(self.dim):
            x_filtered = self.conv(
                x[:, idx].unsqueeze(1),
                weight=self.weights[idx],
                stride=1,
                padding='same',
            )
            x_filtered_list.append(x_filtered)
        return torch.cat(x_filtered_list, dim=1)


class BilateralFilter_OutofMemory(nn.Module):

    def __init__(self,
                 kernel_size,
                 sigma_space=None,
                 sigma_density=None,
                 device='cpu'):
        super().__init__()
        # initialization
        self.kernel_size = tuple(kernel_size)
        self.dim = len(kernel_size)
        self.sigma_space = sigma_space
        self.sigma_density = sigma_density
        if sigma_space is None:
            self.sigma_space = tuple([
                0.3 * ((k_size - 1) * 0.5 - 1) + 0.8 for k_size in kernel_size
            ])
        if sigma_density is None:
            self.sigma_density = 0.8

        self.pad_size = tuple([(ksize - 1) // 2 for ksize in kernel_size])
        self.gaussian_weight = gaussian_kernel(kernel_size).to(device)

    def forward(self, x):
        # [b,c,x,y,z]
        x_pad = F.pad(
            x,
            pad=[xpad for xpad in self.pad_size[::-1] for _ in range(2)],
            mode="replicate",
        )
        x_patches = x_pad.unfold(2, self.kernel_size[0],
                                 1).unfold(3, self.kernel_size[1], 1)
        x_unsqz = x.unsqueeze(-1).unsqueeze(-1)
        if self.dim == 3:
            x_patches = x_patches.unfold(4, self.kernel_size[2], 1)
            x_unsqz = x_unsqz.unsqueeze(-1)

        diff_density = x_patches - x_unsqz
        weight_density = torch.exp(-(diff_density**2) /
                                   (2 * self.sigma_density**2))

        sum_dim = tuple(range(-1, -self.dim - 1, -1))
        weight_density /= weight_density.sum(dim=sum_dim, keepdim=True)

        # Keep same shape with weight_density
        weight_space_dim = (1, self.dim) + self.dim * (1, ) + self.kernel_size
        weight_space = self.gaussian_weight.view(*weight_space_dim)

        # get the final kernel weight
        weight = weight_density * weight_space
        weight_sum = weight.sum(dim=sum_dim)
        x = (weight * x_patches).sum(dim=sum_dim) / weight_sum

        return x


def diff_x(input, r):
    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1:-r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    left = input[:, :, :, r:2 * r + 1]  # 处理边缘的梯度
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:-r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


def diff_z(input, r):
    left = input[:, :, :, :, r:2 * r + 1]  # 处理边缘的梯度
    middle = input[:, :, :, :, 2 * r + 1:] - input[:, :, :, :, :-2 * r - 1]
    right = input[:, :, :, :, -1:] - input[:, :, :, :, -2 * r - 1:-r - 1]

    output = torch.cat([left, middle, right], dim=4)

    return output


class BoxFilter(nn.Module):

    def __init__(self, r):
        super().__init__()

        self.r = r

    def forward(self, x):
        x = diff_x(x.cumsum(dim=2), self.r[0])
        x = diff_y(x.cumsum(dim=3), self.r[1])
        x = diff_z(x.cumsum(dim=4), self.r[2])
        return x


class GuidedFilter(nn.Module):

    def __init__(self, kernel_size, eps=0.1, device='cpu'):
        super(GuidedFilter, self).__init__()

        self.kernel_size = kernel_size
        self.eps = eps
        self.device = device
        self.boxfilter = BoxFilter(kernel_size)

    def forward(self, I, p):
        # x is guided filter
        n_x, c_x, h_x, w_x, d_x = I.shape

        # N
        N = Variable(I.data.new().resize_((1, 1, h_x, w_x, d_x)).fill_(1.0))
        N = self.boxfilter(N)

        # 1. mean_I, mean_p
        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        # 2. var_I, cov_Ip
        var_I = self.boxfilter(I * I) / N - mean_I * mean_I
        cov_Ip = self.boxfilter(I * p) / N - mean_I * mean_p

        # 3. a, b
        A = cov_Ip / (var_I + self.eps)
        b = mean_p - A * mean_I

        # 4. mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        # 5. q
        return mean_A * I + mean_b
