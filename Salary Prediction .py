#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


salary=pd.read_csv('Salary_Data.csv')
X = salary.iloc[:, :1].values
Y = salary.iloc[:, 1].values


# In[5]:


salary


# In[6]:



print(X)


# In[7]:



print(Y)


# In[8]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=.2, random_state=0)


# In[30]:


X_train


# In[31]:


X_test


# In[32]:


Y_train


# In[33]:


Y_test


# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


lm=LinearRegression()


# In[36]:


lm.fit(X_train, Y_train)


# In[42]:





# In[37]:


y_pred= lm.predict(X_test)


# In[38]:


plt.scatter(X_train,Y_train,color='b')

plt.plot(X_train, lm.predict(X_train),color='r')
plt.title('Salary vs Experience')


# In[39]:


plt.scatter(X_train,Y_train, color='b')
plt.scatter(X_test,Y_test,color='g')
plt.plot(X_train,lm.predict(X_train))
plt.title('Train Test Split')


# In[50]:


for i in range(0,len(X_test)):
    print(X_test[i],"  ", int(Y_test[i]))


# In[51]:


for j in range(0, len(X_test)):
    print(X_test[j],"   ",int(y_pred[j]))


# In[58]:


from sklearn.metrics import r2_score
training_data_accuracy=r2_score(Y_test,y_pred, )
print(training_data_accuracy)


# In[ ]:




