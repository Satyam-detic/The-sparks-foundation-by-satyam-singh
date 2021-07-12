#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics - Task 2: Prediction using Unsupervised Machine Learning
# #Prepared by: satyam singh
# Aim: To predict the number of clusters in iris dataset
# Importing all libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# # Reading the dataset
# Importing the dataset

# In[7]:


iris=pd.read_csv('C:\\Users\\asus\\Desktop\\anaconda\\Iris.csv')


# In[8]:


iris.info()


# # Viewing initial rows of the dataset

# In[11]:


iris.head()


# # Computing the K-value and predicting the optimum number of clusters

# In[12]:


x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# # Now we plot the results, which gives us the elbow curve

# In[13]:


plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# # As we can see, there is no significant change when the number of clusters is 3. As a result, the optimum numbers of clusters for the iris dataset is 3.

# In[14]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 700, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# # Now, we can visualize the predicted number of clusters as centroids in the iris dataset

# In[15]:


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'green', label = 'Centroids')
plt.legend()


# # All the predicted clusters are now visualized in the above plot.
