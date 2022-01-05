#!/usr/bin/env python
# coding: utf-8

# # BANK CUSTOMER CHURN PREDICTION USING SUPPORT VECTOR MACHINE (SVM) CLASSIFIER ALGORITHM

# ## 1. IMPORTING THE PYTHON LIBRARIES

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tree
from sklearn.svm import SVC
from sklearn.tree import export_graphviz, export_text
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import graphviz
import pydotplus

from itertools import product

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

# In[10]:


# Converting the Categorical Variables into Numeric One-Hot Encoded Variables for Decision Tree IDE Model Training Purposes

raw_data_pp = pd.get_dummies(raw_data, columns=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])

print("Execution Completed")


# In[11]:


# Verifying the Columns of the Pre-processed Raw Data Frame after Applying One-Hot Encoding Method

raw_data_pp.head()


# In[12]:


# Verifying the Shape of the Pre-processed Raw Data Frame after Applying One-Hot Encoding Method

raw_data_pp.shape


# In[13]:


# Normalising the Continuous Variables Columns to Scale to a Value Between 0 and 1 for Decision Tree IDE Model Training Purposes

norm_scale_features = ['CreditScore', 'Age', 'Balance','EstimatedSalary']

norm_scale = MinMaxScaler()

raw_data_pp[norm_scale_features] = norm_scale.fit_transform(raw_data_pp[norm_scale_features])

print("Scaling is Completed")


# In[14]:


# Verifying all the Columns of the Final Pre-processed Raw Data Frame after Applying the Scaling Method

raw_data_pp.head()


# In[15]:


# Verifying the Shape of the Pre-processed Raw Data Frame after Applying the Scaling Method

raw_data_pp.shape


# ## 6. DATA SPLIT AS TRAIN DATA AND VALIDATION DATA

# In[16]:


# Defining the Input and the Target Vectors for Decision Tree IDE Model Training Purposes

# Input (Independent) Features/Attributes
X = raw_data_pp.drop('Exited', axis=1).values

# Output (Dependent) Target Attribute
y = raw_data_pp['Exited'].values

print("Execution Completed")


# In[17]:


# Verifying the Shape of the Input and the Output Vectors

print("The Input Vector Shape is {}".format(X.shape))
print("The Output Vector Shape is {}".format(y.shape))


# In[18]:


# Splitting the Data Between Train and Validation Data

X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)

print("Execution Completed")


# In[19]:


# Verifying the Shape of the Train and the Validation Data

print("Input Train: {}".format(X_train.shape))
print("Output Train: {}\n".format(y_train.shape))
print("Input Validation: {}".format(X_validate.shape))
print("Output Validation: {}".format(y_validate.shape))


# ## 7. TRAINING THE SUPPORT VECTOR MACHINE (SVM) CLASSIFIER WITH THE DEFAULT PARAMETERS VALUES

# In[20]:


# Creating an Instance of the Support Vector Machine (SVM) Classifier Model with the Default Parameter Values
svm_model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', 
                coef0=0.0, shrinking=True, probability=False, 
                tol=1e-3, cache_size=200, class_weight=None, 
                verbose=False, max_iter=- 1, decision_function_shape='ovr', 
                break_ties=False, random_state=None)

print("Model Training Started.....")

# Training the Support Vector Machine (SVM) Classifier Model
svm_model.fit(X_train, y_train)

print("Model Training Completed.....")


# ## 8. EXTRACTION OF THE FEATURE NAMES

# In[21]:


features = raw_data_pp.drop('Exited', axis=1).columns
feature_names = []
for feature in features:
    feature_names.append(feature)
feature_names


# ## 9. TRAINING VERSUS VALIDATION ACCURACY

# In[22]:


# Accuracy on the Train Data
print("Training Accuracy: ", svm_model.score(X_train, y_train))

# Accuracy on the Validation Data
print("Validation Accuracy: ", svm_model.score(X_validate, y_validate))


# ## 10. VALIDATING THE CLASSIFIER RESULTS ON THE VALIDATION DATA

# In[23]:


# Validating the Classifier Results on the Validation Data

y_validate_pred = svm_model.predict(X_validate)

y_validate_pred


# ## 11. COMPARING THE VALIDATION ACTUALS WITH THE VALIDATION PREDICTIONS

# In[24]:


# Comparing the Validation Predictions with the Validation Actuals for the first 20 Data Instances

# Validation Actuals
print(y_validate[:20])

# Validation Predictions
print(y_validate_pred[:20])


# ## 12. CONFUSION MATRIX BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[25]:


# Defining the Instance of Confusion Matrix
cm_validation_matrix = confusion_matrix(y_validate, y_validate_pred)

print("Execution Completed")


# ## Method 1 : Plotting the Confusion Matrix with Numeric Values using Seaborn heatmap() Function

# In[26]:


# Method 1 : Plotting the Confusion Matrix with Numeric Values using Seaborn heatmap() Function

churn_cm_plot_1 = sns.heatmap(cm_validation_matrix, annot=True)
churn_cm_plot_1


# ## Method 2 : Plotting the Confusion Matrix with Percentage Values using Seaborn heatmap() Function

# In[27]:


# Method 2 : Plotting the Confusion Matrix with Percentage Values Rounded-off to 2 Decimal Places using Seaborn heatmap() Function

churn_cm_plot_2 = sns.heatmap(cm_validation_matrix/np.sum(cm_validation_matrix), annot=True, fmt='0.2%', cmap='plasma')
churn_cm_plot_2


# ## Method 3 : Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function

# In[28]:


# Method 3 : Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function

cm_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

cm_counts = ["{0:0.0f}".format(value) for value in cm_validation_matrix.flatten()]

cm_percentages = ["{0:0.2%}".format(value) for value in cm_validation_matrix.flatten()/np.sum(cm_validation_matrix)]

cm_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cm_names,cm_counts,cm_percentages)]

cm_labels = np.asarray(cm_labels).reshape(2,2)

sns.heatmap(cm_validation_matrix, annot=cm_labels, fmt='', cmap='jet')


# ## 13. CLASSIFICATION REPORT BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[29]:


# Classification Report and Metrics between the Validation Actuals and the Validation Predictions

target_names = ['No Churn', 'Churn']

# Defining the Classification Report for the Validation Data
classification_report_validation = classification_report(y_validate, y_validate_pred, target_names=target_names)

# Displaying the Classification Report
print(classification_report_validation)


# ## 14. INDIVIDUAL CLASSIFIER METRICS BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[30]:


# Individual Classifier Metrics between the Validation Actuals and the Validation Predictions

# Accuracy
churn_accuracy = round((accuracy_score(y_validate, y_validate_pred))*100, 2)

# F1-score
churn_f1_score = round((f1_score(y_validate, y_validate_pred)*100), 2)

# Precision
churn_precision = round((precision_score(y_validate, y_validate_pred)*100), 2)

# Recall
churn_recall = round((recall_score(y_validate, y_validate_pred)*100), 2)

# ROC AUC Score
churn_roc_auc_score = round((roc_auc_score(y_validate, y_validate_pred)*100), 2)

print("Customer Churn Classifier - Accuracy: {}%".format(churn_accuracy))
print("Customer Churn Classifier - F1-Score: {}%".format(churn_f1_score))
print("Customer Churn Classifier - Precision: {}%".format(churn_precision))
print("Customer Churn Classifier - Recall: {}%".format(churn_recall))
print("Customer Churn Classifier - ROC AUC Score: {}%".format(churn_roc_auc_score))


# ## 15. TUNING THE HYPER-PARAMETERS OF THE SUPPORT VECTOR MACHINE (SVM) CLASSIFIER

# ### Creating a For Loop to Tune the Support Vector Machine (SVM) Classifier for the Various Combinations of the Hyper-Parameters

# In[ ]:


# Method 2

# Creating a For Loop to Tune the Support Vector Machine (SVM) Classifier for the Various Combinations of the Hyper-Parameters

# Setting the Values of the Hyper-Parameters to be used for Tuning the Support Vector Machine (SVM) Classifier
C = [1.0, 1.25, 1.50, 1.75, 2.0, 3.0]                             #p1
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']      #p2
degree = [1, 2, 3, 4, 5]                                          #p3
gamma = ['scale', 'auto']                                         #p4
coef0 = [0.0, 0.5, 1.0]                                           #p5
shrinking = True
probability = False
tol = [1e-3, 1e-4, 1e-5, 0.0]                                     #p6
cache_size = [200, 500, 1000]                                     #p7
class_weight = [None, 'balanced']                                 #p8
verbose = False
max_iter = [-1, 10, 20]                                           #p9
decision_function_shape = ['ovr', 'ovo']                          #p10
break_ties = False
random_state = 10  

scenario_id = 0

for p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 in product(C, kernel, degree, gamma, coef0, tol, cache_size, class_weight, max_iter, decision_function_shape):
    
    # Defining the Support Vector Machine (SVM) Classifier Model with its Hyper-Parameters
    svm_tune_model = SVC(C=p1, kernel=p2, degree=p3, gamma=p4, 
                         coef0=p5, shrinking=shrinking, probability=probability, 
                         tol=p6, cache_size=p7, class_weight=p8, 
                         verbose=verbose, max_iter=p9, decision_function_shape=p10, 
                         break_ties=break_ties, random_state=random_state)
    
    # Fitting and Training the Support Vector Machine (SVM) Classifier Model based on its Hyper-Parameters
    svm_tune_model.fit(X_train, y_train)
    
    # Predicting the Classifier on the Validation Data
    y_validate_tune_pred = svm_tune_model.predict(X_validate)
    
    # Calculating the Accuracy
    churn_tune_accuracy = round((accuracy_score(y_validate, y_validate_tune_pred))*100, 2)
    
    # Incrementing the Scenario_ID for Tracking
    scenario_id += 1
    
    # Displaying the Accuracy Metrics for the Various Combinations of the Hyper-Parameters Tuning
    print(" \n Scenario {}: \n C: {}, kernel: {}, degree: {}, gamma: {}, coef0: {}, \n tol: {}, cache_size: {}, class_weight: {}, max_iter: {}, decision_function_shape: {}, \n Classification Accuracy: {}%".format(scenario_id, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, churn_tune_accuracy))
    
    # Defining the Instance of Confusion Matrix
    plot_confusion_matrix(svm_tune_model, X_validate, y_validate)  
    plt.show()
    title = "\n Confusion Matrix {}: \n C: {}, kernel: {}, degree: {}, gamma: {}, coef0: {}, \n tol: {}, cache_size: {}, class_weight: {}, max_iter: {}, decision_function_shape: {}, \n Classification Accuracy: {}%".format(scenario_id, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, churn_tune_accuracy)
    
    '''
    # Defining the Instance of Confusion Matrix
    cm_validation_matrix_tune = confusion_matrix(y_validate, y_validate_tune_pred)
                                                                                                                                    
    # Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function
    cm_names_tune = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    cm_counts_tune = ["{0:0.0f}".format(value) for value in cm_validation_matrix_tune.flatten()]
    cm_percentages_tune = ["{0:0.2%}".format(value) for value in cm_validation_matrix_tune.flatten()/np.sum(cm_validation_matrix_tune)]
    cm_labels_tune = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cm_names_tune, cm_counts_tune, cm_percentages_tune)]
    cm_labels_tune = np.asarray(cm_labels_tune).reshape(2,2)
    plot = sns.heatmap(cm_validation_matrix_tune, annot=cm_labels_tune, fmt='', cmap='jet')                                                                                           
    title = "Confusion Matrix {} - \n C: {}, kernel: {}, degree: {}, gamma: {}, coef0: {}, \n tol: {}, cache_size: {}, class_weight: {}, max_iter: {}, decision_function_shape: {}, \n Classification Accuracy: {}%".format(scenario_id, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, churn_tune_accuracy)
    plot
    '''
    
    


# ## 16. CONFUSION MATRIX - OPTIMIZED MODEL

# ![confusion_matrix_support_vector_machine_classifier.jpg](attachment:40792263-4523-458b-8708-e50a1b5e8a9f.jpg)

# ### As we can see from the above results; the Support Vector Machine (SVM) Classifier Model tuned with the key parameters as C = 1.0, kernel = 'linear', gamma = 'scale' has performed better in the validation data with the accuracy of about 78.9%.
# 
# ### Hence this model can be considered as the Optimized Model for further deployment.

# In[ ]:




