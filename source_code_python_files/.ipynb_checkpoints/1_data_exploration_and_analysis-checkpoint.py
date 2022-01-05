#!/usr/bin/env python
# coding: utf-8

# # BANK CUSTOMER CHURN PREDICTION USING DECISION TREE - DATA EXPLORATION AND ANALYSIS

# ## 1. IMPORTING THE PYTHON LIBRARIES

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import graphviz
import pydotplus

print("Python Libraries Import Completed")


# ## 2. LOADING THE RAW DATA FROM A CSV FILE

# In[2]:


actual_raw_data = pd.read_csv("/Users/vidhyalakshmiparthasarathy/.CMVolumes/Google-Drive-pbvidhya/~~~VP_Data_Science/DS_Real_Time_Projects/Bank_Customers_Churn_Prediction_Using_7_Various_Classification_Algorithms/data/Bank_Churn_Raw_Data.csv")

print("Raw Data Import Completed")


# ## 3. DATA EXPLORATION

# In[3]:


# Verifying the shape of the data

actual_raw_data.shape


# In[4]:


# Displaying the first 5 Rows of Data Instances

actual_raw_data.head()


# In[5]:


# Displaying the last 5 Rows of Data Instances

actual_raw_data.tail()


# In[6]:


# Verifying the Column Names in the Raw Data

actual_raw_data.columns


# In[7]:


# Verifying the Type of the Columns in the Raw Data

actual_raw_data.dtypes


# In[8]:


# Verifying the Null Values in the Raw Data

actual_raw_data.isnull().sum()


# ## 4. DATA VISUALISATION

# In[9]:


# Creating a New Data Frame To Include Only the Relevant Input Independent Variables and the Output Dependent Variable

raw_data = actual_raw_data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                           'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                           'EstimatedSalary', 'Exited']]

raw_data


# In[10]:


# Pair Plot - Visualising the Relationship Between The Variables

raw_data_graph = sns.pairplot(raw_data, hue='Exited', diag_kws={'bw_method':0.2})


# In[11]:


# Count Plot - Visualising the Relationship Between Each OF The Input Independent Variables and the Output Dependent Variable

input_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                 'EstimatedSalary']

for feature in input_features:
    plt.figure()
    feature_count_plot = sns.countplot(x=feature, data=raw_data, hue='Exited', palette="Set3")


# ### Scatter Plot - Visualising the Relationship Between Each OF The Input Independent Variables and the Output Dependent Variable

# In[12]:


# Scatter Plot - Visualising the Relationship Between Each OF The Input Independent Variables and the Output Dependent Variable

input_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                 'EstimatedSalary']

# Input Variable 'CreditScore'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_1 = sns.scatterplot(data=raw_data, x='CreditScore', y=feature, hue='Exited')


# In[13]:


# Input Variable 'Geography'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_2 = sns.scatterplot(data=raw_data, x='Geography', y=feature, hue='Exited')


# In[14]:


# Input Variable 'Gender'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_3 = sns.scatterplot(data=raw_data, x='Gender', y=feature, hue='Exited')


# In[15]:


# Input Variable 'Age'

for feature in input_features:
    plt.figure() 
    feature_scatter_plot_4 = sns.scatterplot(data=raw_data, x='Age', y=feature, hue='Exited')


# In[16]:


# Input Variable 'Tenure'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_5 = sns.scatterplot(data=raw_data, x='Tenure', y=feature, hue='Exited')


# In[17]:


# Input Variable 'Balance'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_6 = sns.scatterplot(data=raw_data, x='Balance', y=feature, hue='Exited')


# In[18]:


# Input Variable 'NumOfProducts'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_7 = sns.scatterplot(data=raw_data, x='NumOfProducts', y=feature, hue='Exited')


# In[19]:


# Input Variable 'HasCrCard'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_8 = sns.scatterplot(data=raw_data, x='HasCrCard', y=feature, hue='Exited')


# In[20]:


# Input Variable 'IsActiveMember'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_9 = sns.scatterplot(data=raw_data, x='IsActiveMember', y=feature, hue='Exited')


# In[21]:


# Input Variable 'EstimatedSalary'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_10 = sns.scatterplot(data=raw_data, x='EstimatedSalary', y=feature, hue='Exited')


# ## 5. DATA PRE-PROCESSING

# In[22]:


# Converting the Categorical Variables into Numeric One-Hot Encoded Variables for Decision Tree CART Model Training Purposes

raw_data_pp = pd.get_dummies(raw_data, columns=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])

print("Execution Completed")


# In[23]:


# Verifying the Columns of the Pre-processed Raw Data Frame after Applying One-Hot Encoding Method

raw_data_pp.head()


# In[24]:


# Verifying the Shape of the Pre-processed Raw Data Frame after Applying One-Hot Encoding Method

raw_data_pp.shape


# In[25]:


# Normalising the Continuous Variables Columns to Scale to a Value Between 0 and 1 for Decision Tree CART Model Training Purposes

norm_scale_features = ['CreditScore', 'Age', 'Balance','EstimatedSalary']

norm_scale = MinMaxScaler()

raw_data_pp[norm_scale_features] = norm_scale.fit_transform(raw_data_pp[norm_scale_features])

print("Scaling is Completed")


# In[26]:


# Verifying all the Columns of the Final Pre-processed Raw Data Frame after Applying the Scaling Method

raw_data_pp.head()


# In[27]:


# Verifying the Shape of the Pre-processed Raw Data Frame after Applying the Scaling Method

raw_data_pp.shape


# ## 6. DATA SPLIT AS TRAIN DATA AND VALIDATION DATA

# In[28]:


# Defining the Input and the Target Vectors for Decision Tree CART Model Training Purposes

# Input (Independent) Features/Attributes
X = raw_data_pp.drop('Exited', axis=1).values

# Output (Dependent) Target Attribute
y = raw_data_pp['Exited'].values

print("Execution Completed")


# In[29]:


# Verifying the Shape of the Input and the Output Vectors

print("The Input Vector Shape is {}".format(X.shape))
print("The Output Vector Shape is {}".format(y.shape))


# In[30]:


# Splitting the Data Between Train and Validation Data

X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)

print("Execution Completed")


# In[31]:


# Verifying the Shape of the Train and the Validation Data

print("Input Train: {}".format(X_train.shape))
print("Output Train: {}\n".format(y_train.shape))
print("Input Validation: {}".format(X_validate.shape))
print("Output Validation: {}".format(y_validate.shape))


# In[ ]:





# In[ ]:




