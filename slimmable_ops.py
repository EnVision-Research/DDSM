from torch import nn


def make_divisible(v, divisible_factor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisible_factor
    new_v = max(min_value, int(v + divisible_factor / 2) // divisible_factor * divisible_factor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisible_factor
    return new_v


class SlimmableConv2d(nn.Conv2d):
    """
    SuperNet Convolution Module.

    automatically adapt to any number of input channels.

    capable of conducting convolutions according to designate width.

    Args:
        us (bool): 是否可以裁剪
            Default: True
        divisible_factor (int): 使out_channel为几的整数倍
            Default: 8
        divisor (bool): 裁剪时channel切割份数
            Default: 8
        linked (int or string or None): 用于裁剪宽度时skip connection的channel对齐的标志
            Default: None
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 slimmable=True,
                 divisible_factor=8,):
        super(SlimmableConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.depthwise = groups == in_channels
        self.out_channels = out_channels
        self.out_channels_max = out_channels
        self.slimmable = slimmable
        self.divisible_factor = divisible_factor
        self.width_mult = 1


    def forward(self, input):
        input_size = input.size()
        in_channels = input_size[1]
        if self.slimmable:
            out_channels = make_divisible(self.out_channels * self.width_mult, self.divisible_factor)
            # depthwise
            self.groups = in_channels if self.depthwise else 1
            out_channels = in_channels if self.depthwise else out_channels
        else:
            out_channels = self.out_channels

        weight = self.weight[:out_channels,:in_channels,:, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride,
                                 self.padding, self.dilation, self.groups)
        return y


class SlimmableBatchNorm2d(nn.BatchNorm2d):
    """
    SuperNet BatchNorm2d Module.

    automatically adapt to any number of input channels.

    capable of conducting batch norm according to designate width.

    record different mean and varience using submodule batch norm layer.

    learn a shared-weight to adapt to different width.

    Args:
        width_mult_list (list[float]): 可接受宽度的列表
            Default: [1, 0.25]
        divisible_factor (int): 使out_channel为几的整数倍
            Default: 8
    """
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, affine=True, track_running_stats=True)
        self.num_features_max = num_features
        self.divisible_factor = kwargs.get('divisible_factor', 8)
        # for tracking performance during training
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=False) for i in [
                make_divisible(self.num_features_max, self.divisible_factor)
                for _ in range(2)
            ]
        ])
        self.bn_setting = None
        self.ignore_model_profiling = True

    def forward(self, input):
        input_size = input.size()
        weight = self.weight
        bias = self.bias
        c = input_size[1]
        self.num_features = c
        if self.bn_setting == 'max':
            y = nn.functional.batch_norm(input, self.bn[0].running_mean[:c],
                                         self.bn[0].running_var[:c],
                                         weight[:c], bias[:c], self.training,
                                         self.momentum, self.eps)
        elif self.bn_setting == 'min':
            y = nn.functional.batch_norm(input, self.bn[1].running_mean[:c],
                                         self.bn[1].running_var[:c],
                                         weight[:c], bias[:c], self.training,
                                         self.momentum, self.eps)
        else:
            # cannot be used when evaluating
            # assert self.training
            y = nn.functional.batch_norm(input, self.running_mean[:c],
                                         self.running_var[:c], weight[:c],
                                         bias[:c], self.training,
                                         self.momentum, self.eps)
        return y


class SlimmableGroupNorm2d(nn.GroupNorm):
    def forward(self, input):
        input_size = input.size()
        c = input_size[1]
        weight = self.weight[:c]
        bias = self.bias[:c]
        return nn.functional.group_norm(
            input, self.num_groups, weight, bias, self.eps)


class SlimmableLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 slimmable=True,
                 divisible_factor=16,):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.out_features = out_features
        self.width_mult = 1
        self.divisible_factor = divisible_factor
        self.slimmable = slimmable

    def forward(self, x):
        input_size = x.size()
        in_features = input_size[1]
        if self.slimmable:
            out_features = make_divisible(self.out_features * self.width_mult, self.divisible_factor)
        else:
            out_features = self.out_features
        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return nn.functional.linear(x, weight, bias)


class SlimmableConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 slimmable=True,
                 divisible_factor=8,):
        super(SlimmableConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size)
        self.out_channels = out_channels
        self.out_channels_max = out_channels
        self.slimmable = slimmable
        self.divisible_factor = divisible_factor
        self.width_mult = 1

    def forward(self, input):
        input_size = input.size()
        in_channels = input_size[1]
        if self.slimmable:
            out_channels = make_divisible(self.out_channels * self.width_mult, self.divisible_factor)
        else:
            out_channels = self.out_channels
        weight = self.weight[:out_channels, :in_channels, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv1d(input, weight, bias, self.stride,
                                 self.padding, self.dilation, self.groups)
        return y

