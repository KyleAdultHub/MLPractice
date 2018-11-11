from sklearn.preprocessing import PolynomialFeatures
import numpy as np


X = np.arange(6).reshape(3, 2)
print(X)

# 设置多项式阶数为２，其他的默认
poly = PolynomialFeatures(2)
X_1 = poly.fit_transform(X)
print(X_1)
# 同时设置交互关系为true
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_2 = poly.fit_transform(X)
print(X_2)
# 取消偏置项
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_3 = poly.fit_transform(X)
print(X_3)
