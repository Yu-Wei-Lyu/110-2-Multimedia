import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

video = cv2.VideoCapture('test_dataset.avi')

if video.isOpened():
    success = True
else:
    success = False
    print('fail to read file !')

frame_index = 1
imgList = []
while(success):
    success, frame = video.read()
    if success == False:
        print('video read end\'s in line %d' % frame_index)
        break
    imgList.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frame_index += 1


with open('label.txt','r') as f:
    sol = []
    file = f.readlines()
    for index in file:
        sol.append(float(index))
imgList = np.array(imgList).reshape((len(imgList), -1))
print(imgList.shape)

X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
    imgList, sol, test_size=0.5)
x_train = np.array(X_train_sk)
y_train = np.array(y_train_sk)
x_test = np.array(X_test_sk)
y_test = np.array(y_test_sk)

qda = KNeighborsClassifier()
qda = qda.fit(x_train, y_train)
y_pred = qda.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = np.diag(cm).sum()/cm.sum()
print('Sklearn confusion_matrix (KNA):\n{}'.format(cm))
print('Sklearn confusion_matrix (KNA,acc):{}'.format(acc))
