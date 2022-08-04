#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


gas=pd.read_csv("D:\\data science\\assignments\\ass-16 Neural networks\\gas_turbines.csv")


# In[3]:


gas


# In[4]:


gas.head()


# In[5]:


gas.info


# In[6]:


gas.isnull().sum()


# In[7]:


gas.corr


# In[9]:


plt.figure(figsize=(10,10))
sns.heatmap(gas.corr(),annot=True,cmap="inferno")


# In[10]:


X=gas.drop('TEY',axis=1)
X


# In[11]:


Y=gas['TEY']
Y


# In[12]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)
x_scaled


# In[13]:


scaled_gasturbines_data=pd.DataFrame(x_scaled,columns=X.columns)
scaled_gasturbines_data


# In[15]:


X_transformed=scaled_gasturbines_data
X_transformed


# In[17]:


X_train,X_test,Y_train,Y_test=train_test_split(X_transformed,Y,test_size=0.20,random_state=123)


# In[18]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[19]:


model=Sequential()
model.add(Dense(units=10,input_dim=10,activation ='relu',kernel_initializer='normal'))
model.add(Dense(units=6,activation='tanh',kernel_initializer='normal'))
model.add(Dense(units=1,activation='relu',kernel_initializer='normal'))


# In[21]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['mse'])


# In[22]:


model.fit(X_train,Y_train, epochs=100, batch_size=20)


# In[23]:


scores = model.evaluate(X_test,Y_test)
print((model.metrics_names[1]))


# # Forest fires

# In[24]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt


# In[28]:


fifo=pd.read_csv("D:\\data science\\assignments\\ass-16 Neural networks\\forestfires.csv")
fifo


# In[29]:


fifo.head()


# In[30]:


fifo.corr


# In[31]:


fifo.shape


# In[33]:


fifo.isnull().sum()


# In[34]:


sns.countplot(x='size_category',data =ff)


# In[35]:


plt.figure(figsize=(20,10))
sns.barplot(x='month',y='temp',data=ff,
            order=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
plt.title('Month Vs Temp')
plt.xlabel('month')
plt.ylabel('temp')


# # Dropping columns which are not required
# 

# In[37]:


fifo=fifo.drop(columns=['dayfri','daymon','daysat','daysun','daythu','daytue','daywed','monthapr',	
                               'monthaug','monthdec','monthfeb','monthjan','monthjul','monthjun','monthmar',
                               'monthmay','monthnov','monthoct','monthsep'],axis=1)


# In[39]:


fifo


# In[40]:


plt.figure(figsize=(10,10))
sns.heatmap(ff.corr(),annot=True,cmap="inferno")
plt.title("HeatMap of Features for the Classes")


# In[41]:


ff["month"].value_counts()


# In[43]:


month_data={'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}
fifo.replace(month_data,inplace=True)


# In[44]:


fifo


# In[46]:


fifo['size_category'].unique()


# In[48]:


fifo


# In[49]:


fifo.size_category.replace(('small','large'),(1,0),inplace=True)


# In[50]:


fifo


# In[51]:


fifo["day"].value_counts()


# In[52]:


day_data={'day':{'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}}
fifo.replace(day_data,inplace=True)


# In[54]:


fifo


# In[55]:



X=fifo.iloc[:,0:11]
X


# In[56]:


Y=fifo["size_category"]
Y


# In[57]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)
x_scaled


# In[59]:


scaled_fifo=pd.DataFrame(x_scaled,columns=X.columns)
scaled_fifo


# In[61]:


x_transformed=scaled_fifo


# In[62]:


x_transformed


# In[64]:


X


# In[65]:


Y


# In[66]:


X_train,X_test,Y_train,Y_test=train_test_split(x_transformed,Y,test_size=0.20,random_state=123)


# In[67]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[69]:


model=Sequential()
model.add(Dense(units=12,input_dim=11,activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=10,activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))


# In[70]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[71]:


model.fit(X_train,Y_train, epochs=100, batch_size=10)


# In[72]:


scores = model.evaluate(X_test,Y_test)
print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




