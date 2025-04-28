
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(2, 4)
        self.output = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

model = MLP()
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
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

with torch.no_grad():
    pred = model(x_data)
    pred_label = (pred >= 0.5).float()
    print(pred_label)
