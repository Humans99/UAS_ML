import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return w*x


print(f'Prediction before training: f(5) = {forward(5).item()}')


learn_rate = 0.01
n_iters = 100

loss = nn.MSELoss()

optim = torch.optim.SGD([w], lr=learn_rate)

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    l.backward()
    optim.step()
    optim.zero_grad()
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}, w = {w}, loss = {l}')

print(f'Prediction before training: f(5) = {forward(5).item()}')


X = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
Y = torch.tensor([[2, 4, 6, 8]], dtype=torch.float32)
n_samples, n_features = X.shape
print(f'sample = {n_samples}, feature = {n_features}')

X_test = torch.tensor(5, dtype=torch.float32)

in_size = n_features
out_size = n_features
model = nn.Linear(in_size, out_size)
print(f'Prediction before training: f(5) = {forward(5).item()}')
