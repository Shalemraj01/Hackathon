#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')


# In[40]:


data = pd.read_csv(r"C:\Users\shale\Downloads\archive (8)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
data


# In[41]:


num_data = data.select_dtypes(['int','float'])
num_data


# In[42]:


obj_data = data.select_dtypes(['object'])
obj_data


# In[43]:


corr = obj_data.corr()

# plotting the heatmap

plt.figure(figsize=(15,6))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# In[44]:


corr = num_data.corr()

# plot the heatmap

plt.figure(figsize=(15, 7))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# In[ ]:





# # GENDER WISE DISTRIBUTION

# In[46]:


#about half of the customers are male and rest are female
data.gender.value_counts()


# In[47]:


dg = (data['gender'].value_counts()*100.0 /len(data)).plot(kind='bar',stacked = True)
dg.yaxis.set_major_formatter(mtick.PercentFormatter())
dg.set_ylabel('% Customers')
dg.set_xlabel('Gender')
dg.set_ylabel('% Customers')
dg.set_title('Gender Distribution')

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in dg.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)


# In[48]:


dg = (data['gender'].value_counts()*100.0 /len(data)).plot(kind='bar',stacked = True)
dg.yaxis.set_major_formatter(mtick.PercentFormatter())
dg.set_ylabel('% Customers')
dg.set_xlabel('Gender')
dg.set_ylabel('% Customers')
dg.set_title('Gender Distribution')

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in dg.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)


# In[ ]:





# In[ ]:





# In[51]:


dg = (data['SeniorCitizen'].value_counts()*100.0 /len(data))\
.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 )                                                                           
dg.yaxis.set_major_formatter(mtick.PercentFormatter())
dg.set_ylabel('Senior Citizens',fontsize = 8)
dg.set_title('% of Senior Citizens', fontsize = 8)


# # Diving deep into the Churn

# In[54]:


dg = (data['Churn'].value_counts()*100.0 /len(data)).plot(kind='bar',stacked = True,figsize = (4,4))
dg.yaxis.set_major_formatter(mtick.PercentFormatter())
dg.set_ylabel('% Customers',size = 14)
dg.set_xlabel('Churn',size = 14)


# In[55]:


#contract type to churn 
dg = pd.crosstab(data['SeniorCitizen'],data['Churn'],margins=True)
dg.drop('All',inplace=True)
dg.drop('All',axis=1,inplace=True)
dg.plot.bar(stacked=True,figsize=(5,5))
plt.show()


# In[ ]:


dg = sns.distplot(data['tenure'], hist=True, kde=False, bins=20, color = 'blue',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
dg.set_ylabel('# of Customers')
dg.set_xlabel('Tenure (months)')
dg.set_title('# of Customers by their tenure')


# In[71]:


ax = data['Contract'].value_counts().plot(kind = 'bar',width = 0.3)
ax.set_ylabel('# of Customers')
ax.set_title('# of Customers by Contract Type')


# In[ ]:





# In[ ]:




