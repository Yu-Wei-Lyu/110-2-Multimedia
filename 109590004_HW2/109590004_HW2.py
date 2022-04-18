from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def getCM(y_test, y_pred):
    dataLen = len(set(y_test))
    cm = np.zeros([dataLen, dataLen],dtype=np.int32)
    for i in range(len(y_pred)):
        cm[y_test[i], y_pred[i]] += 1;
    return cm

def getAcc(cm):
    allData = sum(sum(cm))
    correctData = 0
    for i in range(len(cm)):
        correctData += cm[i, i]
    return correctData / allData


class Gaussian_classifier():
    def ___init__(self):
        self.mu = np.array([])
        self.cov = np.array([])

    def fit(self, data_train, label_train):
        mu, cov = [], []
        for i in range(np.max(label_train)+1):
            pos = np.where(label_train == i)[0]
            tmp_data = data_train[pos, :]
            tmp_cov = np.cov(np.transpose(tmp_data))
            tmp_mu = np.mean(tmp_data, axis=0)
            mu.append(tmp_mu)
            cov.append(tmp_cov)
        self.mu = np.array(mu)
        self.cov = np.array(cov)

    def predict(self, x_test):
        d_value = []
        for tmp_mu, tmp_cov in zip(self.mu, self.cov):
            d = len(tmp_mu)
            zero_center_data = x_test - tmp_mu
            tmp = np.dot(zero_center_data.transpose(), np.linalg.inv(tmp_cov))
            tmp = -0.5*np.dot(tmp, zero_center_data)
            tmp1 = (2 * np.pi)**(-d/2) * np.linalg.det(tmp_cov)**(-0.5)
            tmp = tmp1 * np.exp(tmp)
            d_value.append(tmp)
        d_value = np.array(d_value)

        return np.argmax(d_value), d_value


# use "open" method to read file
with open('iris_x.txt', 'r') as f:
    x = []
    file = f.readlines()
    for index in file:
        data = index.split()
        x.append([float(data[0]), float(data[1]),
                  float(data[2]), float(data[3])])
with open('iris_y.txt', 'r') as f:
    y = []
    file = f.readlines()
    for index in file:
        y.append(int(index))

# use sklearn train_test_split, random state = 20220413
X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
    x, y, test_size=0.2, random_state=20220413)
x_train = np.array(X_train_sk)
y_train = np.array(y_train_sk)
x_test = np.array(X_test_sk)
y_test = np.array(y_test_sk)

# use sklearn to compute MSE
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
mse = np.mean((y_pred - y_test) ** 2)
print('MSE :\n{}'.format(mse))

# Course Learned Quadratic Discriminant Analysis
xqda = Gaussian_classifier()
xqda.fit(x_train, y_train)
y_pred = []
for i in range(len(x_test)):
    y_single_pred, probability = xqda.predict(x_test[i])
    y_pred.append(y_single_pred)
cm = getCM(y_test, y_pred)
acc = getAcc(cm)
print('Course Learned confusion_matrix (QDA):\n{}'.format(cm))
print('Course Learned confusion_matrix (QDA,acc):{}'.format(acc))

# Sklearn Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda = qda.fit(x_train, y_train)
y_pred = qda.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = np.diag(cm).sum()/cm.sum()
print('Sklearn confusion_matrix (QDA):\n{}'.format(cm))
print('Sklearn confusion_matrix (QDA,acc):{}'.format(acc))