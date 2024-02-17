#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv(r"C:\Users\GOLLA\OneDrive\Documents\Documents\Housing (1).csv")


# In[5]:


df.head(10)


# In[6]:


# COLUMNS :

df.columns


# In[7]:


# SIZE OF THE DATASET 

df.shape


# In[8]:


# DATA TYPES OF THE COLUMNS OF THE DATASET

df.info()


# In[9]:


# CHECKING FOR NULL VALUES

df.isnull().sum()


# In[10]:


# REMOVAL OF DUPLICATE VALUE 

counter = 0
rs,cs = df.shape

df.drop_duplicates(inplace=True)

if df.shape==(rs,cs):
    print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
else:
    print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {rs-df.shape[0]}')


# In[11]:


# CONVERTING ALL OUR CATEGORICAL DATA COLUMNS TO NUMERIC FORM

from sklearn.preprocessing import LabelEncoder
categ = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]

# Encode Categorical Columns
le = LabelEncoder()
df[categ] = df[categ].apply(le.fit_transform)


# In[12]:


df.head()


# In[13]:


# CORRELATION BETWEEN THE COLUMNS

corr = df.corr()
plt.figure(figsize=(12,7))
sns.heatmap(corr,cmap='coolwarm',annot=True)


# In[14]:


X = df.drop(['price'],axis=1)
y = df['price']


# In[15]:


X


# In[16]:


y


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[18]:


# LENGTH OF X_train AND X_test

len(X_train),len(X_test)


# In[19]:


# IMPORTING THE MODULE

from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[20]:


# FITTING THE DATA INTO THE MODEL

model.fit(X_train,y_train)


# In[21]:


# PREDICTING THE OUTCOMES

y_predict = model.predict(X_test)


# In[22]:


y_predict


# In[23]:


from sklearn.metrics import r2_score,mean_absolute_error
score = r2_score(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)


# In[24]:


score


# In[25]:


mae


# In[ ]:




