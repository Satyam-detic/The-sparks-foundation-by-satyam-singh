#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Spark Foundation
# # ( Data Science and Business Analytics Intern )
# # Author :Satyam singh
# Exploratory Data Analysis - Retail
# (Level - Beginner)
# 

# In[9]:


import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
import seaborn as sns
import scikitplot as skplt


# In[8]:


get_ipython().system('pip install scikit-plot')


# In[14]:


get_ipython().system('pip install pandas_profiling')


# In[15]:


from pandas_profiling import ProfileReport


# In[17]:


df= pd.read_csv('C:\\Users\\asus\\Desktop\\anaconda\\SampleSuperstore.csv')


# # Analysing Dataset

# In[18]:


df.head()


# In[19]:


df.tail()


# In[20]:


df.ndim


# In[21]:


df.shape


# In[22]:


df.dtypes


# In[23]:


df.columns


# In[24]:


df.isnull().sum()


# In[25]:


df.isna().sum()


# In[26]:


df.isna().any()


# In[27]:


df.info()


# In[28]:


df.describe().transpose()


# # EDA In Jupyter Notebook
# Reverse engineering is performed

# In[29]:


profile= ProfileReport(df, title="EDA", explorative=True)


# In[30]:


profile.to_widgets()


# # EDA in HTML file

# In[31]:


profile.to_file("Output.html")


# In[ ]:




