import numpy as np
import matplotlib.pyplot as plt

# 가상의 데이터 생성
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 편향을 위해 X에 1 추가
X_b = np.c_[np.ones((100, 1)), X]

# 초기값 설정
eta = 0.1  # 학습률
n_iterations = 1000  # 반복 횟수
m = 100  # 샘플 개수

# Gradient Descent (GD) 구현
theta_gd = np.random.randn(2, 1)  # 무작위 초기화
theta_gd_path = []

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    theta_gd = theta_gd - eta * gradients
    theta_gd_path.append(theta_gd)

theta_gd_path = np.array(theta_gd_path)

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.plot(theta_gd_path[:, 0], theta_gd_path[:, 1], "r-s", linewidth=1, label="GD")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.title("Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
