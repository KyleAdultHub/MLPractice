import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# 1.从数据集中载入人脸数据
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
target_names = faces['target_names']
data = faces['data']
target = faces['target']
images = faces['images']
X_train,x_test,y_train,y_true = train_test_split(data,target,test_size=0.1)

# 2.PCA降维
pca = PCA(n_components=0.9, whiten=True)
pca.fit(X_train, y_train)
X_train_pca = pca.transform(X_train)
x_test_pca = pca.transform(x_test)

# 3.创建支持向量机模型
svc = SVC(kernel='rbf')

# 4.参数选择
C = [1, 3, 5, 7, 9]
gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
clf = GridSearchCV(svc, param_grid={'C': C, 'gamma': gamma})
clf.fit(X_train_pca, y_train)
print("best params {}".format(clf.best_params_))

# 5.对测试集进行预测
y_pre = clf.predict(x_test_pca)
clf_score = clf.score(x_test_pca, y_true)

plt.figure(figsize=(12, 20))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(x_test[i].reshape(50, 37), cmap='gray')
    true_name = target_names[y_true[i]].split()[-1]
    predict_name = target_names[y_pre[i]].split()[-1]
    title = 'T:'+true_name+'\nP:'+predict_name
    plt.title(title)
    plt.axis('off')



