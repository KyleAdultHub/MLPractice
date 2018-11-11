from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


def load_small_datasets():
    """
    加载鸢尾花小规模数据集
    :return: None
    """
    X, y = datasets.load_iris(return_X_y=True)
    X_train , X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    ss = StandardScaler()
    X_train_standard = ss.fit_transform(X_train)
    x_test_standard = ss.fit_transform(X_test)
    return X_train_standard, x_test_standard, Y_train, Y_test


X_train, X_test, Y_train, Y_test = load_small_datasets()
lgt = LogisticRegressionCV(multi_class='ovr', fit_intercept=True, cv=2, solver='lbfgs', tol=0.01)
lgt.fit(X_train, Y_train)
print(lgt.predict(X_test))
print(lgt.score(X_test, Y_test))






