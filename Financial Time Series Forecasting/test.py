#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 23:18:40 2017

@author: zwz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:08:44 2017

@author: zwz
"""



# In[2]
import pandas
import matplotlib.pyplot as plt
import numpy
dataframe = pandas.read_csv('AAPL_m_ad_zoo.csv', usecols=[1], engine='python')
dataset = numpy.log(dataframe.values)
dataset = dataset.astype('float32')
dataset
plt.plot(dataset)

# In[3]
look_back =1
train_size = 417 #int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size-look_back-1:len(dataset),:]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[4]:
# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# In[5]:
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(len(trainX),len(testX))

# In[6] SVM:
from sklearn.svm import SVR
import matplotlib.pyplot as plt
dataset_size = len(trainX)
trainX = trainX.reshape(dataset_size,-1)
dataset_size = len(trainY)
trainY = trainY.reshape(dataset_size,-1)
dataset_size = len(testX)
testX = testX.reshape(dataset_size,-1)
dataset_size = len(testY)
testY = testY.reshape(dataset_size,-1)
print(len(trainX),len(testX))
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
model_svr=svr_rbf.fit(trainX,trainY)
y_train_rbf= model_svr.predict(trainX)
y_test_rbf=model_svr.predict(testX)
model_lin=svr_lin.fit(trainX,trainY)
y_train_lin= model_lin.predict(trainX)
y_test_lin= model_lin.predict(testX)
model_poly=svr_poly.fit(trainX,trainY)
y_train_poly= model_poly.predict(trainX)
y_test_poly= model_poly.predict(testX)



## plot

lw = 2
plt.plot(trainY,color='darkorange', label='data')
plt.hold('on')
plt.plot(y_train_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(y_train_lin, color='c', lw=lw, label='Linear model')
plt.plot(y_train_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('time')
plt.ylabel('fitted_train')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
plt.plot(testY,color='darkorange', label='data')
plt.hold('on')
plt.plot(y_test_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(y_test_lin, color='c', lw=lw, label='Linear model')
plt.plot(y_test_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('time')
plt.ylabel('fitted_test')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

# In[7] write data:
f=open('y_train_rbf.txt','w')
for y in y_train_rbf:
    f.write(str(y)+'\n')
f.close()
f=open('y_test_rbf.txt','w')
for y in y_test_rbf:
    f.write(str(y)+'\n')
f.close()

f=open('y_train_lin.txt','w')
for y in y_train_lin:
    f.write(str(y)+'\n')
f.close()
f=open('y_test_lin.txt','w')
for y in y_test_lin:
    f.write(str(y)+'\n')
f.close()
    
f=open('y_train_poly.txt','w')
for y in y_train_poly:
    f.write(str(y)+'\n')
f.close()
f=open('y_test_poly.txt','w')
for y in y_test_poly:
    f.write(str(y)+'\n')
f.close()

## In[8] calculate error:
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
rbf_train_rmse = math.sqrt(mean_squared_error(trainY,y_train_rbf))
rbf_test_rmse= math.sqrt(mean_squared_error(testY,y_test_rbf))
rbf_train_mae = mean_absolute_error(trainY,y_train_rbf)
rbf_test_mae= mean_absolute_error(testY,y_test_rbf)
print('RBF Train: %.4f RMSE' % (rbf_train_rmse),'%.4f MAE' % (rbf_train_mae))
print('RBF Test: %.4f RMSE' % (rbf_test_rmse),'%.4f MAE' % (rbf_test_mae))



lin_train_rmse = math.sqrt(mean_squared_error(trainY,y_train_lin))
lin_test_rmse= math.sqrt(mean_squared_error(testY,y_test_lin))
lin_train_mae = mean_absolute_error(trainY,y_train_lin)
lin_test_mae= mean_absolute_error(testY,y_test_lin)
print('Lin Train: %.4f RMSE' % (lin_train_rmse),'%.4f MAE' % (lin_train_mae))
print('Lin Test: %.4f RMSE' % (lin_test_rmse),'%.4f MAE' % (lin_test_mae))


poly_train_rmse = math.sqrt(mean_squared_error(trainY,y_train_poly))
poly_test_rmse= math.sqrt(mean_squared_error(testY,y_test_poly))
poly_train_mae = mean_absolute_error(trainY,y_train_poly)
poly_test_mae= mean_absolute_error(testY,y_test_poly)
print('Poly Train: %.4f RMSE' % (poly_train_rmse),'%.4f MAE' % (poly_train_mae))
print('Poly Test: %.4f RMSE' % (poly_test_rmse),'%.4f MAE' % (poly_test_mae))



