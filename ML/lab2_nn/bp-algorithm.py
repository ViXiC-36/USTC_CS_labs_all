from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from pandas import DataFrame
import numpy as np

_LR_ = 0.0001
_EPOCHS_ = 10
_HIDDEN_SIZE_ = 100

iris = datasets.load_iris()
df = DataFrame(iris.data, columns=iris.feature_names)
df["target"] = list(iris.target)
X = df.iloc[:, 0:4] # all rows, first 5 columns
Y = df.iloc[:, 4] # all rows, column 5
# 划分数据
"""
TODO在此填入你的代码
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 42 from the official site example

sc = StandardScaler()
sc.fit(X)
standard_train = sc.transform(X_train)
standard_test = sc.transform(X_test)


# 构建 mlp 模型
"""
TODO在此填入你的代码
? hidden_layer_sizes 隐藏层有几个neurons，default=(100,)
? activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
? solver: {‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
? alpha 正则化参数, default=0.0001
? learning_rate: {‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
? learning_rate_init , default=0.001
? max_iter 最大迭代次数
? shuffle 是否在每次迭代时打乱数据, default=True
? random_state 随机种子
? tol 优化的容忍度, default=1e-4
"""
mlp = MLPClassifier(hidden_layer_sizes=(_HIDDEN_SIZE_,), # modified
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    learning_rate='constant',
                    learning_rate_init=_LR_, # modified
                    max_iter=_EPOCHS_,  # modified
                    random_state=42)
# 拟合数据
"""
TODO在此填入你的代码
"""
mlp.fit(standard_train, Y_train)
# 得到预测结果
"""
TODO在此填入你的代码
"""
result = mlp.predict(standard_test)


# 查看模型结果
print("测试集合的 y 值：", list(Y_test))
print("神经网络预测的的 y 值：", list(result))
print("预测的准确率为：", mlp.score(standard_test, Y_test))
print("层数为：", mlp.n_layers_)
print("迭代次数为：", mlp.n_iter_)
print("损失为：", mlp.loss_)
print("激活函数为：", mlp.out_activation_)


# 代码的手动实现
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # 初始化偏置
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):  # sigmoid 计算方式
        """
        TODO在此填入你的代码
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):  # sigmoid 导数计算方式
        """
        TODO在此填入你的代码
        """
        return x * (1 - x)

    def forward(self, X):
        """
        TODO在此填入你的代码
        """
        # input -> hidden
        self.hidden_input = X @ self.weights_input_hidden + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        # hidden -> output
        self.output_input = self.hidden_output @ self.weights_hidden_output + self.bias_output
        self.output = self.sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, output, learning_rate):
        """
        TODO在此填入你的代码
        """
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta @ self.weights_hidden_output.T
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # update the weights and biases
        self.weights_hidden_output += (self.hidden_output.T @ output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += (X.T @ hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(0.5 * (y - output) ** 2)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

    def predict(self, X):
        return np.round(self.forward(X))


# 将标签转换为独热编码
def one_hot_encode(labels):
    num_classes = len(np.unique(labels))
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels


# 构建神经网络
input_size = X_train.shape[1]
hidden_size = _HIDDEN_SIZE_
output_size = len(np.unique(Y_train))  # 根据训练集标签确定输出层大小
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 将标签转换为独热编码
Y_train_encoded = one_hot_encode(Y_train)

# 训练神经网络
print("training.......")
nn.train(standard_train, Y_train_encoded, epochs=_EPOCHS_, learning_rate=_LR_)

# 预测测试集
predictions = nn.predict(standard_test)

# 计算准确率
accuracy = accuracy_score(Y_test, np.argmax(predictions, axis=1))

# 查看模型结果
print("测试集合的 y 值：", list(Y_test))
print("神经网络预测的的 y 值：", list(np.argmax(predictions, axis=1)))
print("预测的准确率为：", accuracy)
