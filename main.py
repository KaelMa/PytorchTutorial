import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print('z=', z)
print('out=', out)

out.backward()
# 输出梯度 d(out)/dx
print(x.grad)