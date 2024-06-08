#!/usr/bin/env python
# coding: utf-8

# # giving permissions to all the libraries
# 
# 

# In[15]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')


# # Let us read the data file

# In[2]:


data = pd.read_csv(r"C:\Users\shale\Downloads\archive (8)\WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[4]:


data


# In[14]:


#finding out the data types in the given data set
data.dtypes


# In[35]:


#converting total charges to a numerical type 
data.TotalCharges = pd.to_numeric(data.TotalCharges,errors='coerce')
data.TotalCharges


# In[63]:


#finding out the missing values
data.isnull().sum()


# In[73]:


#we can see that there are 11 missing values in the totalcharges column lets just drop them
data.dropna(inplace= True)


# In[74]:


data.isnull().sum()


# # lets check how many customers have left the company from the dataset

# In[85]:


data['Churn'].value_counts()

# from the given data set it can be seen that 80% have left the company
# In[127]:


#Get correlation of Churn with other variables


# In[135]:





# In[16]:


#lets differentiate object and numerical data into different tables inorder to do feature engineering


# In[19]:


num_data = data.select_dtypes(['float','int'])
num_data


# In[ ]:


#lets drop the output column and the customerID from the object data


# In[81]:


object_data = data.select_dtypes(['object'])
object_data


# In[ ]:


# lets drop the customerID from the objectdata


# In[82]:


object_data2 = object_data.iloc[:,1:]
object_data2


# 

# In[75]:


#converting predicted variable into numerical variable.
data['Churn'].replace(to_replace='Yes',value= 1,inplace=True)
data['Churn'].replace(to_replace='No',value = 0, inplace= True)


# In[76]:


data.Churn


# In[86]:


#lets make one-hot encoding to the categorical data 


# In[88]:


obj_data_dummies = pd.get_dummies(object_data2)
obj_data_dummies


# In[102]:


y = data.Churn
y


# In[103]:


x = obj_data_dummies


# In[104]:


x


# In[105]:


#scaling all the variable from 0 to 1 by using normalization method


# In[108]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax


# In[109]:


#fit transformation by using min_maxscaler will give the values between min and max range
norm1 = x.apply(lambda x:minmax.fit_transform(x.values.reshape(-1,1)).ravel())
norm1


# In[110]:


#creating test and train dataset
from sklearn.model_selection import train_test_split


# In[111]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state= 10)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


# In[147]:


#importing the logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)


# In[148]:


from sklearn import metrics
model = LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)


# In[150]:


print (metrics.accuracy_score(y_test,y_predict))


# In[151]:


print (metrics.precision_score(y_test,y_predict))


# In[152]:


print (metrics.recall_score(y_test,y_predict))


# In[154]:


print (metrics.confusion_matrix(y_test,y_predict))


# In[156]:


lr_score = lr.score(x_test,y_test)
lr_score


# In[157]:


print(metrics.classification_report(y_test,y_predict))


# In[144]:


# To get the weights of all the variables
weights = pd.Series(model.coef_[0],index=x.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# In[124]:


print(weights.sort_values(ascending = False)[-10:].plot(kind='bar'))


# In[146]:


# AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
# n_estimators = 50 (default value) 
# base_estimator = DecisionTreeClassifier (default value)
model.fit(x_train,y_train)
preds = model.predict(x_test)
metrics.accuracy_score(y_test, preds)


# In[ ]:




