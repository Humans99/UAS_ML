import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0


def forward(x):
    return w*X


def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()


print(f'Prediction before training: f(5) = {forward(5)}')

learn_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)

    dw = gradient(X, Y, y_pred)
    w -= learn_rate*dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w}, loss = {l}')

print(f'Prediction after training: f(5) = {forward(5)}')
