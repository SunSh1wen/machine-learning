from sklearn.datasets import load_iris


def datasets_demo():
    iris = load_iris()
    print("查看数据集描述： ", iris.DESCR)

if __name__ == '__main__':
    datasets_demo()



