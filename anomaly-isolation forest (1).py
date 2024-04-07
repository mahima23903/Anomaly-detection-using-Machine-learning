#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv(r'processedweather.csv')
df


# In[3]:


df=df.drop(columns='Unnamed: 0')


# In[4]:


df.dtypes
df['time'] = pd.to_datetime(df['time'],dayfirst= True)
df
df_=df


# In[5]:


df= df.set_index('time')


# In[6]:


model=IsolationForest(n_estimators=50, max_samples='auto', contamination='auto',max_features=1.0)


# In[7]:


model.fit(df[['prcp']])


# In[8]:


df['scores']=model.decision_function(df[['prcp']])
df['anomaly']=model.predict(df[['prcp']])
anomaly = model.decision_function(df[['prcp']])
df


# In[ ]:





# In[9]:


df_=df_.set_index('time')


# In[10]:


model.fit(df_[['tmax']])


# In[11]:


df_['scores_temp']=model.decision_function(df_[['tmax']])
df_['anomaly_temp']=model.predict(df_[['tmax']])
df_


# In[12]:


outliers_counter = len(df[df['prcp'] > 0.4])
outliers_counter


# In[13]:


print("Accuracy percentage:", list(df['anomaly']).count(-1)/(outliers_counter))


# In[14]:


outliers_counter_ = len(df_[df_['tavg'] < 25])
outliers_counter_


# In[15]:


print("Accuracy percentage:", list(df_['anomaly_temp']).count(-1)/(outliers_counter_))


# In[ ]:





# In[ ]:




