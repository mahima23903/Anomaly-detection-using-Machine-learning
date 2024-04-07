#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


import pandas as pd
import sklearn as sk
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc

import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv(r"processedweather.csv")


# In[4]:


df
df = df.drop(columns='Unnamed: 0')


# In[5]:


df['targettemp'] = df.shift(-1)['tmax']
df


# In[6]:


df= df.iloc[:-1:].copy()  #iloc- to access specified integer values of the dataframe - indexing dataframe
df=df.set_index('time')
df


# In[7]:


#Ridge regression to predict future temperature
from sklearn.linear_model import Ridge

r= Ridge(alpha=.1)


# In[8]:


predictors=['tavg','tmin','tmax','prcp']


# In[9]:


x=df[predictors]
y=df['targettemp']


# In[10]:


#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=42)


# In[11]:


r.fit(x_train,y_train)


# In[12]:


predictions=r.predict(x_test)


# In[13]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
error = mean_absolute_error(y_test,predictions)
error


# In[14]:


combined = pd.concat([y_test, pd.Series(predictions, index=y_test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[15]:


combined.tail(15)


# In[16]:


combined.head(25).plot()


# In[17]:


r.coef_   #finding effect of each variable on target variable


# In[18]:


df = df.iloc[30:,:].copy()


# In[19]:


def create_predictions(predictors, df, r):
    x = df[predictors]
    y = df['targettemp']
    x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.2, random_state=42)

    r.fit(x_train, y_train)
    predictions = r.predict(x_test)

    error = mean_squared_error(y_test, predictions)
    
    combined = pd.concat([y_test, pd.Series(predictions, index=y_test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


# In[20]:


r.coef_


# In[21]:


df.corr()["targettemp"]


# In[22]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[23]:


combined.sort_values("diff", ascending=False)


# In[24]:


# precipitation

df['tprcp'] = df.shift(-1)['prcp']
df
df= df.iloc[:-1:].copy()  #iloc- to access specified integer values of the dataframe - indexing dataframe
df


# In[25]:


x_=df[predictors]
y_=df['tprcp']


# In[26]:


x_train_, x_test_, y_train_, y_test_ = train_test_split(x_,y_ , test_size=0.2, random_state=42)


# In[27]:


r.fit(x_train_,y_train_)


# In[28]:


predictions_=r.predict(x_test_)


# In[29]:


error_ = mean_absolute_error(y_test_,predictions_)
error_


# In[30]:


combined_ = pd.concat([y_test_, pd.Series(predictions_, index=y_test_.index)], axis=1)
combined_.columns = ["actual_prcp", "predictions_prcp"]
combined_


# In[31]:


combined_["diff_"] = (combined_["actual_prcp"] - combined_["predictions_prcp"]).abs()
combined_


# In[32]:


df["month_max"] = df["tmax"].rolling(30).mean()

df["month_day_max"] = df["month_max"] / df["tmax"]

df["max_min"] = df["tmax"] /df["tmin"]


# In[35]:


combined_.head(15)


# In[ ]:




