import torch

from torch.autograd import Function

# Inherit from Function
class LinearFunction(Function):
    # 创建torch.autograd.Function类的一个子类
    # 必须是staticmethod
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    # ctx在这里类似self，ctx的属性可以在backward中调用。
    # 自己定义的Function中的forward()方法，所有的Variable参数将会转成tensor！因此这里的input也是tensor．在传入forward前，autograd engine会自动将Variable unpack成Tensor。
    def forward(ctx, input, weight, bias=None):
        print(type(input))
        ctx.save_for_backward(input, weight, bias) # 将Tensor转变为Variable保存到ctx中
        output = input.mm(weight.t())  # torch.t()方法，对2D tensor进行转置
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output为反向传播上一级计算得到的梯度值
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        # 分别代表输入,权值,偏置三者的梯度
        # 判断三者对应的Variable是否需要进行反向求导计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight) # 复合函数求导，链式法则
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)# 复合函数求导，链式法则
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias