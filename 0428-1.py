import torch

class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

x_data = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_data = torch.tensor([[0.], [1.], [1.], [0.]])

model = Perceptron()
learning_rate = 0.1

for epoch in range(5000):
    pred = model(x_data)
    loss = mse_loss(pred, y_data)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    if (epoch+1) % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 결과 확인
with torch.no_grad():
    pred = model(x_data)
    pred_label = (pred > 0.5).float()
    print(pred_label)