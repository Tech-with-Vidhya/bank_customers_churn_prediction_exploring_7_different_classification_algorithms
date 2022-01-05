#!/usr/bin/env python
# coding: utf-8

# # BANK CUSTOMER CHURN PREDICTION USING ENSEMBLE HIST GRADIENT BOOSTING CLASSIFIER ALGORITHM

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import export_graphviz, export_text
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import graphviz
import pydotplus

from itertools import product

import math

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


# Converting the Categorical Variables into Numeric One-Hot Encoded Variables for Decision Tree IDE Model Training Purposes

raw_data_pp = pd.get_dummies(raw_data, columns=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])

print("Execution Completed")


# In[23]:


# Verifying the Columns of the Pre-processed Raw Data Frame after Applying One-Hot Encoding Method

raw_data_pp.head()


# In[24]:


# Verifying the Shape of the Pre-processed Raw Data Frame after Applying One-Hot Encoding Method

raw_data_pp.shape


# In[25]:


# Normalising the Continuous Variables Columns to Scale to a Value Between 0 and 1 for Decision Tree IDE Model Training Purposes

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


# Defining the Input and the Target Vectors for Decision Tree IDE Model Training Purposes

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


# ## ENSEMBLE HIST GRADIENT BOOSTING CLASSIFIER ALGORITHM

# ## 7. TRAINING THE ENSEMBLE - HIST GRADIENT BOOSTING CLASSIFIER ALGORITHM

# In[32]:


# Defining the Parameters of the HistGradientBoostingClassifier Model

# Default Hyper-Parameters Values

loss = 'binary_crossentropy'
learning_rate = 0.1
max_iter = 100
max_leaf_nodes = 31
max_depth = None
min_samples_leaf = 20 
l2_regularization = 0.0
max_bins = 255
categorical_features = None
monotonic_cst = None
warm_start = False
early_stopping = True
scoring = 'accuracy'
validation_fraction = 0.1
n_iter_no_change = 10
tol = 1e-7
verbose = 0
random_state = None


# Creating an Instance of the HistGradientBoostingClassifier Model
hist_gradient_boosting_model = HistGradientBoostingClassifier(loss = loss,
                                                              learning_rate = learning_rate, 
                                                              max_iter = max_iter,
                                                              max_leaf_nodes = max_leaf_nodes,
                                                              max_depth = max_depth,
                                                              min_samples_leaf = min_samples_leaf,
                                                              l2_regularization = l2_regularization,
                                                              max_bins = max_bins,
                                                              categorical_features = categorical_features,
                                                              monotonic_cst = monotonic_cst,
                                                              warm_start = warm_start,
                                                              early_stopping = early_stopping,
                                                              scoring = scoring,
                                                              validation_fraction = validation_fraction,
                                                              n_iter_no_change = n_iter_no_change,
                                                              tol = tol,
                                                              verbose = verbose,
                                                              random_state = random_state)

print("Model Training Started.....")

# Training the HistGradientBoostingClassifier Model
hist_gradient_boosting_model.fit(X_train, y_train)

print("Model Training Completed.....")


# ## 8. VERIFYING THE TOTAL ITERATIONS AND TOTAL DECISION TREES PER ITERATION

# In[33]:


# Verifying the Total Number of Iterations Executed and the Total Number of Decision Trees Created for Each Iteration 
# with "early_stopping" hyper-parameter set to "True"

print("Actual Total Number of Iterations: ", max_iter)
print("Total Number of Iterations Executed: ", hist_gradient_boosting_model.n_iter_)
print("Total Number of Decision Trees Created per Iteration: ", hist_gradient_boosting_model.n_trees_per_iteration_)


# ## 9. CALCULATING AND COMPARING THE TRAINING ACCURACY AND VALIDATION ACCURACY

# In[34]:


# Calculating the Training and the Validation Accuracy for Each Iteration

training_accuracy_array = hist_gradient_boosting_model.train_score_
validation_accuracy_array = hist_gradient_boosting_model.validation_score_

# Creating an Empty DataFrame to Hold the Training and the Validation Accuracy Scores for all the Iterations
scores_final_df = pd.DataFrame()

# Creating an Iteration Counter Variable and Initializing it to Zero
iteration_counter = 0

for train_accuracy, validation_accuracy in zip(training_accuracy_array, validation_accuracy_array):
    print("Training Accuracy: {}, Validation Accuracy: {}".format(train_accuracy, validation_accuracy))
    train_accuracy_percentage = train_accuracy * 100
    validation_accuracy_percentage = validation_accuracy * 100
    #print("Training Accuracy in %: {}, Validation Accuracy in %: {}".format(train_accuracy_percentage, validation_accuracy_percentage))
    
    # Accuracy on the Train Data
    #print("Training Accuracy: ", train_accuracy)

    # Accuracy on the Validation Data
    #print("Validation Accuracy: ", validation_accuracy)
    
    # Incrementing the iteration_counter by 1 for Each Iteration
    iteration_counter += 1
    
    # Creating an Individual DataFrame to Hold the Training and the Validation Accuracy for Each Iteration
    scores_df = pd.DataFrame({"Iteration_ID": [iteration_counter], "Training_Accuracy": [train_accuracy], "Validation_Accuracy": [validation_accuracy], "Training_Accuracy_Percentage": [train_accuracy_percentage], "Validation_Accuracy_Percentage": [validation_accuracy_percentage]})
    
    # Concatenating the Individual Results DataFrame with the Final DataFrame
    scores_final_df = pd.concat([scores_final_df, scores_df], axis=0, ignore_index=True)
 
scores_final_df


# ## 10. VISUALIZING THE TRAINING ACCURACY AND VALIDATION ACCURACY

# In[35]:


# Visualizing the Training Accuracy and the Validation Accuracy

plt.plot(scores_final_df['Iteration_ID'], scores_final_df['Training_Accuracy_Percentage'], label='Training_Accuracy_Percentage')
plt.plot(scores_final_df['Iteration_ID'], scores_final_df['Validation_Accuracy_Percentage'], label='Validation_Accuracy_Percentage')
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Iteration versus Accuracy in %")
#plt.legend()
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.figure(figsize=(40, 20))
plt.show()


# ## 11. VALIDATING THE CLASSIFIER RESULTS ON THE VALIDATION DATA

# In[36]:


# Validating the Classifier Results on the Validation Data

y_validate_pred = hist_gradient_boosting_model.predict(X_validate)

y_validate_pred


# In[37]:


# Validating the Classifier Probability Results on the Validation Data

y_validate_prob_pred = hist_gradient_boosting_model.predict_proba(X_validate)

y_validate_prob_pred


# ## 12. COMPARING THE VALIDATION ACTUALS WITH THE VALIDATION PREDICTIONS

# In[38]:


# Comparing the Validation Predictions with the Validation Actuals for the first 20 Data Instances

# Validation Actuals
print(y_validate[:20])

# Validation Predictions
print(y_validate_pred[:20])


# ## 13. CONFUSION MATRIX BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[39]:


# Defining the Instance of Confusion Matrix
cm_validation_matrix = confusion_matrix(y_validate, y_validate_pred)

print("Execution Completed")


# ## Method 1 : Plotting the Confusion Matrix with Numeric Values using Seaborn heatmap() Function

# In[40]:


# Method 1 : Plotting the Confusion Matrix with Numeric Values using Seaborn heatmap() Function

churn_cm_plot_1 = sns.heatmap(cm_validation_matrix, annot=True)
churn_cm_plot_1


# ## Method 2 : Plotting the Confusion Matrix with Percentage Values using Seaborn heatmap() Function

# In[41]:


# Method 2 : Plotting the Confusion Matrix with Percentage Values Rounded-off to 2 Decimal Places using Seaborn heatmap() Function

churn_cm_plot_2 = sns.heatmap(cm_validation_matrix/np.sum(cm_validation_matrix), annot=True, fmt='0.2%', cmap='plasma')
churn_cm_plot_2


# ## Method 3 : Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function

# In[42]:


# Method 3 : Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function

cm_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

cm_counts = ["{0:0.0f}".format(value) for value in cm_validation_matrix.flatten()]

cm_percentages = ["{0:0.2%}".format(value) for value in cm_validation_matrix.flatten()/np.sum(cm_validation_matrix)]

cm_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cm_names,cm_counts,cm_percentages)]

cm_labels = np.asarray(cm_labels).reshape(2,2)

sns.heatmap(cm_validation_matrix, annot=cm_labels, fmt='', cmap='jet')


# ## 14. CLASSIFICATION REPORT BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[43]:


# Classification Report and Metrics between the Validation Actuals and the Validation Predictions

target_names = ['No Churn', 'Churn']

# Defining the Classification Report for the Validation Data
classification_report_validation = classification_report(y_validate, y_validate_pred, target_names=target_names)

# Displaying the Classification Report
print(classification_report_validation)


# ## 15. INDIVIDUAL CLASSIFIER METRICS BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[44]:


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


# ## 16. TUNING THE HYPER-PARAMETERS OF THE HIST GRADIENT BOOSTING CLASSIFIER MODEL

# In[ ]:


# Method

# Creating an Empty DataFrame to Hold the Training and the Validation Accuracy Scores for all the Iterations
tune_model_scores_df_final = pd.DataFrame()

# Creating an Empty Pandas DataFrame to Hold the Tuned Model Results for various combinations of the Hyper-Parameters
hist_gradient_boosting_tune_model_df_final = pd.DataFrame()

# Setting the Values of the Hyper-Parameters to be used for Tuning the HistGradientBoostingClassifier Model
loss = 'binary_crossentropy'
learning_rate = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]                          #p1
max_iter = [10, 25, 50, 100, 150, 200, 250]                                        #p2
max_leaf_nodes = [5, 11, 21, 31, 41]                                               #p3
max_depth = [None, 3, 5, 7, 10]                                                    #p4
min_samples_leaf = [5, 10, 20, 30]                                                 #p5
l2_regularization = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]                                 #p6
max_bins = [5, 10, 15, 20, 255]                                                    #p7
categorical_features = [None]                                                      #p8
monotonic_cst = [None]                                                             #p9
warm_start = False
early_stopping = True
scoring = ['accuracy', 'roc_auc']                                                  #p10
validation_fraction = [None, 0.1, 0.2, 0.3]                                        #p11
n_iter_no_change = [0, 1, 2, 3, 5, 10, 15, 20]                                     #p12
tol = [1, 0.75, 0.5, 0.25, 0, 1e-3, 1e-5, 1e-7, 1e-10, 1e-12, 1e-15, 1e-20]        #p13
verbose = 99
random_state = [None, 10]                                                          #p14    
    
# Creating a Scenario ID Counter Variable to Track the Various Scenarios Related to the Hyper-Parameters Tuning    
scenario_id = 0

# Creating a For Loop to Tune the HistGradientBoostingClassifier Model for the Various Combinations of the Hyper-Parameters

for p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 in product(learning_rate, max_iter, max_leaf_nodes, max_depth, 
                                                                           min_samples_leaf, l2_regularization, max_bins, 
                                                                           categorical_features, monotonic_cst, scoring, 
                                                                           validation_fraction, n_iter_no_change, tol,
                                                                           random_state):
    
    # Defining an Instance of the HistGradientBoostingClassifier Model
    hist_gradient_boosting_tune_model = HistGradientBoostingClassifier(loss = loss,
                                                                      learning_rate = p1, 
                                                                      max_iter = p2,
                                                                      max_leaf_nodes = p3,
                                                                      max_depth = p4,
                                                                      min_samples_leaf = p5,
                                                                      l2_regularization = p6,
                                                                      max_bins = p7,
                                                                      categorical_features = p8,
                                                                      monotonic_cst = p9,
                                                                      warm_start = warm_start,
                                                                      early_stopping = early_stopping,
                                                                      scoring = p10,
                                                                      validation_fraction = p11,
                                                                      n_iter_no_change = p12,
                                                                      tol = p13,
                                                                      verbose = verbose,
                                                                      random_state = p14)
    
    # Fitting and Training the AdaBoostClassifier Model based on its Hyper-Parameters
    hist_gradient_boosting_tune_model.fit(X_train, y_train)
    
    # Predicting the Classifier on the Validation Data
    y_validate_tune_pred = hist_gradient_boosting_tune_model.predict(X_validate)
    
    # Calculating the Accuracy
    accuracy_hist_gradient_boost_churn_tune = round((accuracy_score(y_validate, y_validate_tune_pred))*100, 2)
    
    # F1-score
    f1_score_hist_gradient_boost_churn_tune = round((f1_score(y_validate, y_validate_pred)*100), 2)

    # Precision
    precision_hist_gradient_boost_churn_tune = round((precision_score(y_validate, y_validate_pred)*100), 2)

    # Recall
    recall_hist_gradient_boost_churn_tune = round((recall_score(y_validate, y_validate_pred)*100), 2)
    
    # ROC AUC Score
    roc_auc_score_hist_gradient_boost_churn_tune = round((roc_auc_score(y_validate, y_validate_pred)*100), 2)
    
    # Incrementing the Scenario_ID for Tracking
    scenario_id += 1
    
    # Displaying the Accuracy Metrics for the Various Combinations of the Hyper-Parameters Tuning
    print(" \n Scenario {} - learning_rate: {}, max_iter: {}, max_leaf_nodes: {}, max_depth: {}, min_samples_leaf: {}, \n l2_regularization: {}, max_bins: {}, categorical_features: {}, monotonic_cst: {}, scoring: {}, \n validation_fraction: {}, n_iter_no_change: {}, tol: {}, random_state: {}, \n\n Hist Gradient Boosting Classification Accuracy: {}%, \n Hist Gradient Boosting Classification F1-Score: {}%, \n Hist Gradient Boosting Classification Precision: {}%, \n Hist Gradient Boosting Classification Recall: {}%, \n Hist Gradient Boosting classification ROC AUC Score: {}% \n".format(scenario_id, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, accuracy_hist_gradient_boost_churn_tune, f1_score_hist_gradient_boost_churn_tune, precision_hist_gradient_boost_churn_tune, recall_hist_gradient_boost_churn_tune, roc_auc_score_hist_gradient_boost_churn_tune))
    
    # Defining the Instance of Confusion Matrix
    plot_confusion_matrix(hist_gradient_boosting_tune_model, X_validate, y_validate)  
    plt.show()
    #title = "Confusion Matrix {} - learning_rate: {}, max_iter: {}, max_leaf_nodes: {}, max_depth: {}, min_samples_leaf: {}, l2_regularization: {}, max_bins: {}, categorical_features: {}, monotonic_cst: {}, scoring: {}, validation_fraction: {}, n_iter_no_change: {}, tol: {}, random_state: {}, \n Hist Gradient Boosting Classification Accuracy: {}%, Hist Gradient Boosting Classification F1-Score: {}%, Hist Gradient Boosting Classification Precision: {}%, \n Hist Gradient Boosting Classification Recall: {}%, Hist Gradient Boosting classification ROC AUC Score: {}%".format(scenario_id, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, accuracy_hist_gradient_boost_churn_tune, f1_score_hist_gradient_boost_churn_tune, precision_hist_gradient_boost_churn_tune, recall_hist_gradient_boost_churn_tune, roc_auc_score_hist_gradient_boost_churn_tune))
    
    # Defining a Individual DataFrame to Hold the Hyper-Parameters Values and their Corresponding Performance Metrics Scores
    hist_gradient_boosting_tune_model_df = pd.DataFrame({'Scenario_id': [scenario_id], 'learning_rate': [p1], 'max_iter': [p2], 'max_leaf_nodes': [p3], 'max_depth': [p4], 'min_samples_leaf': [p5], 'l2_regularization': [p6], 'max_bins': [p7], 'categorical_features': [p8], 'monotonic_cst': [p9], 'scoring': [p10], 'validation_fraction': [p11], 'n_iter_no_change': [p12], 'tol': [p13], 'random_state': [p14], 'Accuracy': [accuracy_hist_gradient_boost_churn_tune], 'F1-Score': [f1_score_hist_gradient_boost_churn_tune], 'Precision': [precision_hist_gradient_boost_churn_tune], 'Recall': [recall_hist_gradient_boost_churn_tune], 'ROC_AUC_Score': [roc_auc_score_hist_gradient_boost_churn_tune]})
    
    # Concatenating the Individual Results DataFrame with the Final DataFrame
    hist_gradient_boosting_tune_model_df_final = pd.concat([hist_gradient_boosting_tune_model_df_final, hist_gradient_boosting_tune_model_df], axis=0, ignore_index=True)
    
    # Creating an Iteration Counter Variable and Initializing it to Zero
    iteration_counter_tune = 0
    
    # Calculating the Training and the Validation Accuracy for Each Iteration
    training_accuracy_array_tune = hist_gradient_boosting_tune_model.train_score_
    validation_accuracy_array_tune = hist_gradient_boosting_tune_model.validation_score_
    
    for train_accuracy_tune, validation_accuracy_tune in zip(training_accuracy_array_tune, validation_accuracy_array_tune):
        #print("Training Accuracy: {}, Validation Accuracy: {}".format(train_accuracy_tune, validation_accuracy_tune))
        train_accuracy_tune_percentage = train_accuracy_tune * 100
        validation_accuracy_tune_percentage = validation_accuracy_tune * 100
        #print("Training Accuracy in %: {}, Validation Accuracy in %: {}".format(train_accuracy_tune_percentage, validation_accuracy_tune_percentage))
    
        # Incrementing the iteration_counter by 1 for Each Iteration
        iteration_counter_tune += 1
    
        # Creating an Individual DataFrame to Hold the Training and the Validation Accuracy for Each Iteration Based on the Scenario
        tune_model_scores_df = pd.DataFrame({"Scenario_id": [scenario_id], "Iteration_ID": [iteration_counter_tune], "Training_Accuracy": [train_accuracy_tune], "Validation_Accuracy": [validation_accuracy_tune], "Training_Accuracy_Percentage": [train_accuracy_tune_percentage], "Validation_Accuracy_Percentage": [validation_accuracy_tune_percentage]})
    
        # Concatenating the Individual Results DataFrame with the Final DataFrame
        tune_model_scores_df_final = pd.concat([tune_model_scores_df_final, tune_model_scores_df], axis=0, ignore_index=True)
    
    
# Sorting the Final DataFrame Based on the Increasing Value of Accuracy
hist_gradient_boosting_tune_model_df_final_sorted = hist_gradient_boosting_tune_model_df_final.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    
print("Model Tuning Execution Completed")    


# In[ ]:


# Displaying the DataFrame Containing the Various Performance Metrics Results of the Various Combinations of the Hyper-Parameters Tuning  

#hist_gradient_boosting_tune_model_df_final_sorted


# In[ ]:


# Displaying the DataFrame Containing the Various Training and Validation Acuuracy Scores for Each Scenario Iteration-wise

#tune_model_scores_df_final


# ## 18. CONCLUSION

# ### As we can see from the above results; the better performing optimized Ensemble Hist Gradient Boosting Classifier Models are with the below hyper-parameters:
# > ### loss = 'binary_crossentropy'
# > ### learning_rate = 0.1
# > ### max_iter = 100
# > ### max_leaf_nodes = 31
# > ### max_depth = None
# > ### min_samples_leaf = 20
# > ### l2_regularization = 0.0
# > ### max_bins = 255
# > ### categorical_features = None
# > ### monotonic_cst = None
# > ### warm_start = False
# > ### early_stopping = True
# > ### scoring = 'accuracy'
# > ### validation_fraction = 0.1
# > ### n_iter_no_change = 10
# > ### tol = 1e-7
# > ### verbose = 0
# > ### random_state = None
# 
# ### Performance Metrics and Outcomes for the above mentioned better performing optimized Ensemble Adaptive Boost Classifier Models are:
# > ### Classification Accuracy = 86.0%
# > ### Classification F1-Score = 58.33%
# > ### Classification Precision = 78.4%
# > ### Classification Recall = 46.45%
# > ### Classification ROC AUC Score = 71.51%

# In[ ]:




