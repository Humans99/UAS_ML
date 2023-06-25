import numpy as np
import torch

x = torch.empty(1)
print(x)
x = torch.rand(5, 3)
print(x)
x = torch.ones(2, 2, dtype=torch.float16)
print(x.size)
x = torch.tensor([2.5, 0.1])
print(x)


x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = torch.sub(x, y)
print(z)

x = torch.rand(3, 2)
print(x)
print(x[:, 0])
print(x[1, 1])
print(x[1, :])

a = np.ones(4)
b = torch.from_numpy(a)
print(a)
print(b)


if torch.cuda.is_available():
    dev = torch.device("cuda")
    y = torch.ones_like(x, device=dev)
    x = x.to(dev)
    z = x + y
    z.to("cpu")
    print(z)
