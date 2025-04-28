import torch

# 퍼셉트론 모델 정의
class Perceptron1D(torch.nn.Module):
    def __init__(self):
        super(Perceptron1D, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 입력 1, 출력 1

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid 활성화 함수 사용

# 손실 함수 (MSE)
def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

# 학습 데이터
x_data = torch.tensor([[0.], [1.]])  # 입력이 1개
y_data = torch.tensor([[0.], [1.]])  # 출력이 입력과 동일하게

# 모델 생성
model = Perceptron1D()
learning_rate = 0.1

# 학습
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

# 결과 확인
with torch.no_grad():
    pred = model(x_data)
    pred_label = (pred >= 0.5).float()
    print(f"예측 결과:\n{pred_label}")
    print(f"정답:\n{y_data}")