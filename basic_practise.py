
"""

@Author: Luocai
@file: basic_practise.py
@time: 2019/05/06 20:51
@desc:


"""
import torch

#############
#  Tensors  #
#############

# 创建一个未初始化的 5*3 矩阵
x = torch.empty(5, 3)
print(x)
# 创建一个随机初始化的 5*3 矩阵
rand_x = torch.rand(5, 3)
print(rand_x)
# 创建一个数值为0，类型为 long 的矩阵
zero_x = torch.zeros(5, 3, dtype=torch.long)
print(zero_x)
# 创建一个 tensor，初始化方式是当前数值
tensor1 = torch.tensor([5.5, 3])
print(tensor1)
# 根据已存在的 tensor 创建新的 tensor
tensor2 = tensor1.new_ones(5, 3, dtype=torch.double)  # new_* 方法需要输入 tensor 大小
print(tensor2)
# 修改数值类型，但有相同的 size
tensor3 = torch.randn_like(tensor2, dtype=torch.float)
print('tensor3: ', tensor3)
# get size
print(tensor3.size())

################
#  Operations  #
################

# 加法操作
tensor4 = torch.rand(5, 3)
print('tensor3 + tensor4= ', tensor3 + tensor4)
print('tensor3 + tensor4= ', torch.add(tensor3, tensor4))
# 给定加法结果的 tensor 变量
result = torch.empty(5, 3)
torch.add(tensor3, tensor4, out=result)
print('add result= ', result)
# 直接修改 tensor 变量
tensor3.add_(tensor4)
print('tensor3= ', tensor3)
# 可以用类似 numpy 的索引操作来访问 tensor 的某一维, 比如访问 tensor3 第一列数据
print(tensor3[:, 0])

# 当 CUDA 可用的时候，可运行这段代码，采用 torch.device() 来改变 tensors 是否在 GPU 上进行计算操作
if torch.cuda.is_available():
    device = torch.device("cuda")          # 定义一个 CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 显示创建在 GPU 上的一个 tensor
    x = x.to(device)                       # 也可以采用 .to("cuda")
    z = x + y
    print("----", z)
    print(z.to("cpu", torch.double))       # .to() 方法也可以改变数值类型
