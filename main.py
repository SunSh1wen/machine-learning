from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def datasets_demo():
    iris = load_iris()
    # print("鸢尾花数据集：/n", iris)
    # print("查看数据集描述： ", iris.DESCR)
    print("查看特征值的名字:", iris.feature_names)
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print(x_train)
    return None


def dict_demo():
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 90},
            {'city': '深圳', 'temperature': 10}]
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None


if __name__ == '__main__':
    # datasets_demo()
    dict_demo()
