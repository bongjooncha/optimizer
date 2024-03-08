import numpy as np

# 가상의 데이터 생성
np.random.seed(42)      

np.random.seed(42)      #다른 GD와 비교하기 위해서 seed 정의

X = 2 * np.random.rand(100, 1)        
y = 4 + 3 * X + np.random.randn(100, 1)

# 4*1임. 4는 변수를 곱하지 않는 상수이므로 1을 넣어 놓는다.(= 편향을 위해 1 생성)
X_b = np.c_[np.ones((100, 1)), X]
theta_gd = np.random.randn(2, 1)
X_b.dot(theta_gd)

print(X_b.dot(theta_gd)-y)


X_b = np.c_[np.ones((100, 1)), X]

# 초기값 설정
eta = 0.1                   # 학습률
n_iterations = 1000         # 반복 횟수(1epoch에서 반복되는 회수)
m = 100                     # 샘플 개수 100개

theta_gd = np.random.randn(2, 1)  # 세타값을 무작위로 생성
theta_bgd = theta_gd
theta_gd_path = []                # 앞으로 생성될 세타 쌍을 저장하는 공간
theta_bgd_path=[]

for iteration in range(n_iterations):  # 1회 반복 당 시행되는 과정
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    theta_gd = theta_gd - eta * gradients
    theta_gd_path.append(theta_gd)

for iteration in range(n_iterations):
    for i in range(m):
        gradients = 2/m * X_b[i, :].T.dot(X_b[i, :].dot(theta_bgd) - y[i])
        theta_bgd = theta_bgd - eta * gradients
        theta_bgd_path.append(theta_bgd)

# if theta_gd_path != theta_bgd_path:
#     print('다름')
print(X_b)

# theta_gd_path = np.array(theta_gd_path)
# print(theta_gd_path)