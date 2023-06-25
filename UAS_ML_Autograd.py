import torch

x = torch.randn(3, requires_grad=True)
y = x + 2
z = y*y*2

print(x)
print(y)
print(z.mean())

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
print(x.grad)

with torch.no_grad():
    y = x+2
    print(y)


weights = torch.ones(4, requires_grad=True)

for epoch in range(5):
    model_output = (weights*3).sum()

    model_output.backward()
    print(weights.grad)

