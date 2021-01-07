# GMIT HDip Data Analytics 2020; Machine Learning and Statistics project
# Author : Nigel Slack ;  G00376340

# Python Flask server to provide power output predictions for a wind turbine based on wind speed inputs taken from a web page.


# Within neural networks a model may have insufficient capacity to learn the dataset, or it may have too much capacity 
# and 'memorize' the dataset, possibly then getting stuck during optimization [1]. The capacity is determined by the number 
# of nodes and layers.

# As we only have one input variable we don't need to analyse correlations between different inputs, and the plot of the data shows
# there are no outliers that may significantly skew results, thus requiring standardising the data, other than the zero values that 
# are between the minimum and maximum inputs that yield non-zero outputs. Two predictions are made, one that includes these
# values and one that excludes them, as both may be useful to a user.

# Different authors suggest that experimentation and intuition should be used to determine the best configuration of nodes and layers 
# rather than there being clearly defined steps to obtain anoptimal model [1][2][3].

# Number of nodes  Number of Layers  Number of Epochs  Loss  Sum sqrt((Actual - Predicted)**2)/50
# 1                1                 100               3890  9.47
# 1                1                 500               3929  7.42            
# 10               1                 100               2486  6.74    
# 10               1                 500                145  1.53  
# 50               1                 100                666  3.09   
# 50               1                 500                 27  0.76   
# 100              1                 500                 39  0.99            
# 50               1                 700                 23  0.69            
# 50               1                 800                 19  0.62            
# 50               1                1000                 18  0.59            
# 7 / 7            2                1000                 17  0.52  (loss stopped declining after around 900 epochs)          


# [1] https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/
# [2] https://www.datacamp.com/community/tutorials/deep-learning-python
# [3] https://towardsdatascience.com/optimizing-neural-networks-where-to-start-5a2ed38c8345

# Get the external libraries for reading and pre-processing the input data, applying 
import tensorflow.keras as kr
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import flask as fl
from flask import request

# Get the contents of the csv file
df = pd.read_csv('powerproduction.txt', sep=",")

# Make sure the values are sorted by the first column (speed), to make it easy to find the minimum and maxiumum wind speed values
# below and above which the power output is zero.[1]
# [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
df.sort_values(by=df.columns[0])

# Get the contents of the two columns, speed and power
# We'll use all the dataset values (excluding ones where the power output is zero at the start and end of the dataset, and 
# excluding the values between the min and max where the power output is also zero for a second dataset). In the Jupyter notebook
# we trained the neural network and obtained an 'sklearn' regression line using part of the dataset, to test the two methods 
# with the remaining values, but using all the values will give us the best models for predicting power outputs for a user.
s = df["speed"].to_numpy()
po = df["power"].to_numpy()

# Find the minimum speed value below which all power outputs are zero, and the maximum value above which the power output is zero
# Also find all the entries where the speed is between the minimum and maximum values and the power output is still zero - 
# turbine down time. 
i = 0
ix = []
ix2 = []
minP = 0
maxP = 0
minS = 0
maxS = 0

# The entries are in speed order, so find the largest speed value at the start of the dataset with a power output of zero
for px in po:
    if px == 0:
# Reset the minimum wind speed value    
      if s[i] > minS:      
        minS = s[i]
    else:
        break
    i+=1

# Get the indices of all the dataset entries where the power output is zero
i = 0
for px in po:
    if px == 0:
        ix.append(i)
    else:
        maxP = s[i]
        maxI = i
        if s[i] > maxS:
# Find the maximum speed for which the power output is not zero        
            maxS = s[i]
        if minP == 0:
            minP = s[i]
            minI = i
    i+=1

# minI is the index for which all entries in the dataset below it have zero power output
# maxI is the index for which all entries in the dataset above it have zero power output
# Get the indices of all the dataset entries below the minimum speed and above the maximum speed for which the power output is zero
for i in ix:
    if i < minI or i > maxI:
       ix2.append(i) 

# Create a dataset without the zero power output values at the start and end of the original data values    
s2   = np.delete(s, ix2)
po2  = np.delete(po, ix2)
df2  = df.drop(ix2,axis=0)

# Create a dataset with no zero power output values (ie excluding values where the turbine(s) was offline) 
s  = np.delete(s, ix)
po = np.delete(po, ix)
df = df.drop(ix,axis=0)

# We'll build two neural networks, training the first with the data that excludes the zero values (associated with turbine down time),
# and training the second with the dataset that includes these values.


# We have one dimensional input and output variables, and consequently the Sequential model is the most appropriate to use [1]
# https://keras.io/guides/sequential_model/
nnmodel1 = kr.models.Sequential()
# We'll use the Dense layer type, the most comonly used one, where each neuron is connected to every neuron in the preceeding layer,
# and that applies the formula 'output = activation(dot(input, kernel) + bias)' where 'kernel' is the matrix of weights applied to 
# connections between neurons - the model type discussed in lectures. We only have one column in our input data, causing our 
# input shape (tensor dimension) to be (1,) [1].  
# We'll use the 'sigmoid' activation function - we know there is a non-linear relationship between our input and output (speed/power)
# variables, and with sigmoid being non-linear it enables our network to 'learn' about this relationship. [2] Other activation functions 
# could also be used, to return similar levels of accuracy - they may require the dataset to be passed through the network a different
# number of times (the number of 'epochs') to get to a steady state.
# Setting initial values for the weights (that connect the neurons) to 'glorot_uniform', should provide a start point for efficient, stable 
# processing (a distribution centred around zero, with low variance) [3], [4]. Different distributions for initialising bias values are of
# less importance, with little difference between options. We'll set them both to glorot_uniform.
# [1] https://datascience.stackexchange.com/questions/53609/how-to-determine-input-shape-in-keras
# [2] https://towardsdatascience.com/exploring-activation-functions-for-neural-networks-73498da59b02
# [3] https://deeplizard.com/learn/video/8krd5qKVw-Q
# [4] https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
nnmodel1.add(kr.layers.Dense(7, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel1.add(kr.layers.Dense(7, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel1.add(kr.layers.Dense(1, activation='linear', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
# Here we make minor changes to the weights each time the values are passed through the network to try to minimise the mean squared error 
# (the mean of the square of the difference between actual and predicted values produced by passing the input values through the network
# 'epochs' number of times) [5]. 'Adam' has been shown to be one of the most efficient algorithms for this process [6].
# [5] https://www.tutorialspoint.com/keras/keras_model_compilation.htm
# [6] https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
nnmodel1.compile(kr.optimizers.Adam(lr=0.001), loss='mean_squared_error')
nnmodel1.fit(df['speed'], df['power'], epochs=900, batch_size=10)

nnmodel2 = kr.models.Sequential()
nnmodel2.add(kr.layers.Dense(50, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel2.add(kr.layers.Dense(1, activation='linear', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel2.compile(kr.optimizers.Adam(lr=0.001), loss='mean_squared_error')
nnmodel2.fit(df2['speed'], df2['power'], epochs=500, batch_size=10)

# Use 'sklearn' to fit a regression line to the data. In the Jupyter notebook plots were made of a linear function, a 3rd order polynomial
# and a fifth order polynomial, with the latter providing the best fit to the data [1].
# [1] Polynomial regression ; https://scikit-learn.org/stable/modules/linear_model.html

def f(x,p):
    return p[0] + x*p[1]

def predict(s):
    return f(s,p)

# Create two polynomial models of order 5, and then train them using 'fit'. This expects a 2D array, so reshape the 1D speed values 
# to make them acceptable for 'fit'. [1]
# [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html    
s = s.reshape(-1,1)
s2 = s2.reshape(-1,1)

# Find a polynomial that describes well the transformation that converts input values to output values, then fit a 
# regression line to it. Using 'Pipeline' we can do this in one command [1]. Do this for the two datasets - one including turbine 
# downtime values, and one excluding them.
# [1] Polynomial Regression ; https://ggbaker.ca/data-science/content/ml.html
skmodel1 = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression(fit_intercept=False))])
skmodel2 = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression(fit_intercept=False))])

skmodel1.fit(s,po)
skmodel2.fit(s2,po2)

# flask for web app.
# From Machine Learning and Statistics lecture notes
# Create a new web app.
app = fl.Flask(__name__)

# Add root route. 
@app.route("/")
def home():
  return app.send_static_file('getPower.html')

# Respond to a request from the webpage to get the predicted power output associated with an input wind speed.
# Four values are output, two each from the neural network model and the sklearn regression model, one including turbine downtime and
# one excluding it.
@app.route('/api/speed',methods=['GET', 'POST'])
def speed():
# Get the wind speed value input by the user
  data = request.form.get('name', '')
  dataFloat = float(data)
# If the input value is less than the minimum wind speed or greater than the maximum wind speed, in which cases power output is always 
# zero, return zero for all values.
  if (dataFloat < minS) or (dataFloat > maxS):
    q1 = 0
    q2 = 0
    q3 = 0
    q4 = 0
  else: 
# Get the predicted power values from the two models  
# Create a numpy array from the input value to pass to the neural network model (the expected input type)
    dnp = np.float32(dataFloat)
    npa = np.array( [dnp,] )  
    q1 = nnmodel1.predict( npa)
    q2 = nnmodel2.predict( npa)
# Put the wind speed into a 2D array to pass to the sklearn model (again, the required input data type)
    arr = dnp.reshape(-1,1)
    q3 = skmodel1.predict(arr)
    q4 = skmodel2.predict(arr)

# Pass the resulting predictions back to the webpage
  result = []
  result.append("Power excluding downtime - Model 1 : " + str(int(q1)) + " KwH ; Model 2 : " + str(int(q3)) + " KwH ")
  result.append("Power including downtime - Model 1 : " + str(int(q2)) + " KwH ; Model 2 : " + str(int(q4)) + " KwH ")
  return {"value": result}  
  
# When the webpage loads it will request the minimum and maximum wind speeds from the server below and above which the power output
# is always zero - to tell the user what these values are
@app.route('/api/minmax')
def minmax():
  result = []
  result.append("Min - " + str(minS) + " Km/H")
  result.append("   Max - " + str(maxS) + " Km/H")
  return {"value": result}    