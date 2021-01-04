import tensorflow.keras as kr
import numpy as np
import pandas as pd

df = pd.read_csv('powerproduction.txt', sep=",")
df.sort_values(by=df.columns[0])

s = df["speed"].to_numpy()
po = df["power"].to_numpy()

i = 0
ix = []
ix2 = []
minP = 0
maxP = 0
minS = 0
maxS = 0

for px in po:
    if px == 0:
      if s[i] > minS:
        minS = s[i]
    else:
        break
    i+=1

i = 0
for px in po:
    if px == 0:
        ix.append(i)
    else:
        maxP = s[i]
        maxI = i
        if s[i] > maxS:
            maxS = s[i]
        if minP == 0:
            minP = s[i]
            minI = i
    i+=1

for i in ix:
    if i < minI or i > maxI:
       ix2.append(i) 
    
s2   = np.delete(s, ix2)
po2  = np.delete(po, ix2)
df2  = df.drop(ix2,axis=0)

s  = np.delete(s, ix)
po = np.delete(po, ix)
df = df.drop(ix,axis=0)

nnmodel1 = kr.models.Sequential()
nnmodel1.add(kr.layers.Dense(7, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel1.add(kr.layers.Dense(8, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel1.add(kr.layers.Dense(1, activation='linear', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel1.compile(kr.optimizers.Adam(lr=0.001), loss='mean_squared_error')
nnmodel1.fit(df['speed'], df['power'], epochs=500, batch_size=10)

nnmodel2 = kr.models.Sequential()
nnmodel2.add(kr.layers.Dense(7, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel2.add(kr.layers.Dense(8, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel2.add(kr.layers.Dense(1, activation='linear', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
nnmodel2.compile(kr.optimizers.Adam(lr=0.001), loss='mean_squared_error')
nnmodel2.fit(df2['speed'], df2['power'], epochs=500, batch_size=10)

#testActual = df.to_numpy()

# Polynomial regression ; https://scikit-learn.org/stable/modules/linear_model.html
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as lin

def f(x,p):
    return p[0] + x*p[1]

def predict(s):
    return f(s,p)
    
s = s.reshape(-1,1)
s2 = s2.reshape(-1,1)
skmodel1 = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression(fit_intercept=False))])
skmodel2 = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression(fit_intercept=False))])

skmodel1.fit(s,po)
skmodel2.fit(s2,po2)
#r = model2.score(s,po)

# flask for web app.
import flask as fl
from flask import request

# Create a new web app.
app = fl.Flask(__name__)

# Add root route.
@app.route("/")
def home():
  return app.send_static_file('getPower.html')

@app.route('/api/speed',methods=['GET', 'POST'])
def speed():
  data = request.form.get('name', '')
  dataFloat = float(data)
  if (dataFloat < minS) or (dataFloat > maxS):
    q1 = 0
    q2 = 0
    q3 = 0
    q4 = 0
  else: 
    dnp = np.float32(dataFloat)
    npa = np.array( [dnp,] )  
    q1 = nnmodel1.predict( npa)
    q2 = nnmodel2.predict( npa)
    arr = dnp.reshape(-1,1)
    q3 = skmodel1.predict(arr)
    q4 = skmodel2.predict(arr)

  result = []
  result.append("Power excluding downtime - Model 1 : " + str(int(q1)) + " KwH ; Model 2 : " + str(int(q3)) + " KwH ")
  result.append("Power including downtime - Model 1 : " + str(int(q2)) + " KwH ; Model 2 : " + str(int(q4)) + " KwH ")
  return {"value": result}  
  
@app.route('/api/minmax')
def minmax():
  result = []
  result.append("Min - " + str(minS) + " Km/H")
  result.append("   Max - " + str(maxS) + " Km/H")
  return {"value": result}    