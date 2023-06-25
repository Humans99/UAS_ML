import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)


z = torch.tensor(1.0, requires_grad=True)

y_hat = z * y
loss = (y_hat - y) ** 2
print(loss)

#backward

loss.backward()
print(z.grad)
