
# coding: utf-8

# In[1]:

import pandas
import matplotlib.pyplot as plt
dataset = pandas.read_csv('AAPL_m_ad_zoo.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()


# In[2]:

len(dataset)


# In[3]:

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[4]:

# fix random seed for reproducibility
numpy.random.seed(7)


# In[5]:

# load the dataset
dataframe = pandas.read_csv('AAPL_m_ad_zoo.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset = numpy.log(dataset)


# In[6]:

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[7]:

# split into train and test sets
look_back = 1
train_size = 417 #int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size-look_back-1:len(dataset),:]
print(len(train), len(test))


# In[8]:

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[9]:

# reshape into X=t and Y=t+1

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(len(trainX),len(testX))


# In[10]:

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[11]:

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, verbose=2)


# In[12]:

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
trainScore2 = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Train Score: %.4f RMSE' % (trainScore))
print('Train Score: %.4f MAE' % (trainScore2))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
testScore2 = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.4f RMSE' % (testScore))
print('Test Score: %.4f MAE' % (testScore2))


# In[ ]:

numpy.savetxt("testPredict7.csv", testPredict, delimiter=",")
numpy.savetxt("trainPredict7.csv", trainPredict, delimiter=",")


# In[13]:

get_ipython().magic('matplotlib notebook')
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back)-1:len(dataset), :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

