def softmax(x):
    exp_x = torch.exp(x - x.max(dim=1, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=1, keepdim=True)

def cross_entropy(pred, target):
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1. - eps)
    return (-(target * torch.log(pred)).sum(dim=1)).mean()

class MLP_Multi(torch.nn.Module):
    def __init__(self):
        super(MLP_Multi, self).__init__()
        self.hidden = torch.nn.Linear(2, 6)
        self.output = torch.nn.Linear(6, 3)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x

# 다중 분류 데이터
x_data = torch.tensor([
    [1, 0], [0, 1], [1, 1],
    [0, 0], [1, 0.5], [0.5, 1]
])
y_data = torch.tensor([0, 1, 2, 0, 2, 1])
y_onehot = torch.zeros(len(y_data), 3)
y_onehot.scatter_(1, y_data.view(-1, 1), 1)

model = MLP_Multi()
learning_rate = 0.1
loss_list = []

for epoch in range(5000):
    logits = model(x_data)
    prob = softmax(logits)
    loss = cross_entropy(prob, y_onehot)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    loss_list.append(loss.item())

    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# 예측 결과
with torch.no_grad():
    logits = model(x_data)
    pred = softmax(logits)
    pred_label = torch.argmax(pred, dim=1)
    print(f"예측 결과:\n{pred_label}")
    print(f"정답:\n{y_data}")