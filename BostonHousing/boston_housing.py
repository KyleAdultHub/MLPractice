from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def linear_regression():
    # 可以直接通过接口获取数据
    data_source = datasets.load_boston()
    # 对于数据进行切分 训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data_source.data, data_source.target, test_size=0.2, random_state=1)
    # 标准化的处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 梯度下降算法api
    lrg = LinearRegression()
    # 通过梯度下降的方式 进行迭代式的学习获取一个最小误差对应的模型
    lrg.fit(x_train, y_train)
    y_pre = lrg.predict(x_test)
    # 获取损失的均方误差大小
    ret = mean_squared_error(y_test, y_pre)
    print("梯度下架预测的房价为: ", y_pre)
    print("梯度下降得到的均方误差为: ", ret)
    print("梯度下降的回归系数: ", lrg.coef_)


if __name__ == '__main__':
    linear_regression()