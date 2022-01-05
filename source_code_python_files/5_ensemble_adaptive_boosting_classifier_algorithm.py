#!/usr/bin/env python
# coding: utf-8

# # BANK CUSTOMER CHURN PREDICTION USING ENSEMBLE ADAPTIVE BOOSTING CLASSIFIER ALGORITHM

# ## 1. IMPORTING THE PYTHON LIBRARIES

# In[258]:


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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import export_graphviz, export_text
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import graphviz
import pydotplus

from itertools import product

print("Python Libraries Import Completed")


# ## 2. LOADING THE RAW DATA FROM A CSV FILE

# In[4]:


actual_raw_data = pd.read_csv("/Users/vidhyalakshmiparthasarathy/.CMVolumes/Google-Drive-pbvidhya/~~~VP_Data_Science/DS_Real_Time_Projects/Bank_Customers_Churn_Prediction_Using_7_Various_Classification_Algorithms/data/Bank_Churn_Raw_Data.csv")

print("Raw Data Import Completed")


# ## 3. DATA EXPLORATION

# In[5]:


# Verifying the shape of the data

actual_raw_data.shape


# In[6]:


# Displaying the first 5 Rows of Data Instances

actual_raw_data.head()


# In[7]:


# Displaying the last 5 Rows of Data Instances

actual_raw_data.tail()


# In[8]:


# Verifying the Column Names in the Raw Data

actual_raw_data.columns


# In[9]:


# Verifying the Type of the Columns in the Raw Data

actual_raw_data.dtypes


# In[10]:


# Verifying the Null Values in the Raw Data

actual_raw_data.isnull().sum()


# ## 4. DATA VISUALISATION

# In[11]:


# Creating a New Data Frame To Include Only the Relevant Input Independent Variables and the Output Dependent Variable

raw_data = actual_raw_data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                           'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                           'EstimatedSalary', 'Exited']]

raw_data


# In[12]:


# Pair Plot - Visualising the Relationship Between The Variables

raw_data_graph = sns.pairplot(raw_data, hue='Exited', diag_kws={'bw_method':0.2})


# In[13]:


# Count Plot - Visualising the Relationship Between Each OF The Input Independent Variables and the Output Dependent Variable

input_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                 'EstimatedSalary']

for feature in input_features:
    plt.figure()
    feature_count_plot = sns.countplot(x=feature, data=raw_data, hue='Exited', palette="Set3")


# ### Scatter Plot - Visualising the Relationship Between Each OF The Input Independent Variables and the Output Dependent Variable

# In[14]:


# Scatter Plot - Visualising the Relationship Between Each OF The Input Independent Variables and the Output Dependent Variable

input_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                 'EstimatedSalary']

# Input Variable 'CreditScore'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_1 = sns.scatterplot(data=raw_data, x='CreditScore', y=feature, hue='Exited')


# In[15]:


# Input Variable 'Geography'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_2 = sns.scatterplot(data=raw_data, x='Geography', y=feature, hue='Exited')


# In[16]:


# Input Variable 'Gender'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_3 = sns.scatterplot(data=raw_data, x='Gender', y=feature, hue='Exited')


# In[17]:


# Input Variable 'Age'

for feature in input_features:
    plt.figure() 
    feature_scatter_plot_4 = sns.scatterplot(data=raw_data, x='Age', y=feature, hue='Exited')


# In[18]:


# Input Variable 'Tenure'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_5 = sns.scatterplot(data=raw_data, x='Tenure', y=feature, hue='Exited')


# In[19]:


# Input Variable 'Balance'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_6 = sns.scatterplot(data=raw_data, x='Balance', y=feature, hue='Exited')


# In[20]:


# Input Variable 'NumOfProducts'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_7 = sns.scatterplot(data=raw_data, x='NumOfProducts', y=feature, hue='Exited')


# In[21]:


# Input Variable 'HasCrCard'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_8 = sns.scatterplot(data=raw_data, x='HasCrCard', y=feature, hue='Exited')


# In[22]:


# Input Variable 'IsActiveMember'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_9 = sns.scatterplot(data=raw_data, x='IsActiveMember', y=feature, hue='Exited')


# In[23]:


# Input Variable 'EstimatedSalary'

for feature in input_features:
    plt.figure()
    feature_scatter_plot_10 = sns.scatterplot(data=raw_data, x='EstimatedSalary', y=feature, hue='Exited')


# ## 5. DATA PRE-PROCESSING

# In[24]:


# Converting the Categorical Variables into Numeric One-Hot Encoded Variables for Decision Tree IDE Model Training Purposes

raw_data_pp = pd.get_dummies(raw_data, columns=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])

print("Execution Completed")


# In[25]:


# Verifying the Columns of the Pre-processed Raw Data Frame after Applying One-Hot Encoding Method

raw_data_pp.head()


# In[26]:


# Verifying the Shape of the Pre-processed Raw Data Frame after Applying One-Hot Encoding Method

raw_data_pp.shape


# In[27]:


# Normalising the Continuous Variables Columns to Scale to a Value Between 0 and 1 for Decision Tree IDE Model Training Purposes

norm_scale_features = ['CreditScore', 'Age', 'Balance','EstimatedSalary']

norm_scale = MinMaxScaler()

raw_data_pp[norm_scale_features] = norm_scale.fit_transform(raw_data_pp[norm_scale_features])

print("Scaling is Completed")


# In[28]:


# Verifying all the Columns of the Final Pre-processed Raw Data Frame after Applying the Scaling Method

raw_data_pp.head()


# In[29]:


# Verifying the Shape of the Pre-processed Raw Data Frame after Applying the Scaling Method

raw_data_pp.shape


# ## 6. DATA SPLIT AS TRAIN DATA AND VALIDATION DATA

# In[30]:


# Defining the Input and the Target Vectors for Decision Tree IDE Model Training Purposes

# Input (Independent) Features/Attributes
X = raw_data_pp.drop('Exited', axis=1).values

# Output (Dependent) Target Attribute
y = raw_data_pp['Exited'].values

print("Execution Completed")


# In[31]:


# Verifying the Shape of the Input and the Output Vectors

print("The Input Vector Shape is {}".format(X.shape))
print("The Output Vector Shape is {}".format(y.shape))


# In[32]:


# Splitting the Data Between Train and Validation Data

X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)

print("Execution Completed")


# In[33]:


# Verifying the Shape of the Train and the Validation Data

print("Input Train: {}".format(X_train.shape))
print("Output Train: {}\n".format(y_train.shape))
print("Input Validation: {}".format(X_validate.shape))
print("Output Validation: {}".format(y_validate.shape))


# ## ADAPTIVE BOOSTING ENSEMBLE CLASSIFIER ALGORITHM

# ## 7. TRAINING THE ENSEMBLE - ADAPTIVE BOOSTING CLASSIFIER ALGORITHM

# In[46]:


# Defining the Parameters of the AdaBoostClassifier Model

base_estimator = DecisionTreeClassifier(max_depth=1, criterion='gini')
n_estimators = 50
learning_rate = 1.0
algorithm= 'SAMME.R'
random_state= None


# Creating an Instance of the AdaBoostClassifier Model
ada_boost_model = AdaBoostClassifier(base_estimator=base_estimator,
                                     n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     algorithm=algorithm,
                                     random_state=random_state)

print("Model Training Started.....")

# Training the AdaBoostClassifier Model
ada_boost_model.fit(X_train, y_train)

print("Model Training Completed.....")


# ## 8. DECISION STUMP GRAPHICAL REPRESENTATION AND VISUALISATION

# ### Method 1 : Visualising the Decision Stump using export_graphviz() Function

# In[218]:


# Method 1 : Visualising the Decision Stump Using export_graphviz() Function

# Displaying the First Decision Stump at Index Position 0 in the Base Estimator

# Defining the Decision Stump Graph Data
graph_data_graphviz = tree.export_graphviz(ada_boost_model.estimators_[0], out_file=None, 
                                  feature_names=raw_data_pp.drop('Exited', axis=1).columns,
                                  class_names=raw_data_pp['Exited'].unique().astype(str),
                                  filled=True, rounded=True, special_characters=True,
                                  impurity=True)

#graph_data_graphviz

# Creating the Decision Stump for the Above Graph Data using Graphviz
decision_tree_graph = graphviz.Source(graph_data_graphviz)

# Visualising the Decision Stump
decision_tree_graph


# ### Method 2 : Visualising the Decision Stump using graph_from_dot_data() Function

# In[219]:


# Method 2 : Visualising the Decision Stump Using graph_from_dot_data() Function

# Displaying the Second Decision Stump at Index Position 1 in the Base Estimator

# Defining the Decision Stump Graph Data
graph_data_pydot = tree.export_graphviz(ada_boost_model.estimators_[1], out_file=None, 
                                  feature_names=raw_data_pp.drop('Exited', axis=1).columns,
                                  class_names=raw_data_pp['Exited'].unique().astype(str),
                                  filled=True, rounded=True, special_characters=True,
                                  impurity=True)

#graph_data_pydot


# Creating the Decision Stump for the Above Graph Data using pydotplus
pydot_graph = pydotplus.graph_from_dot_data(graph_data_pydot)
pydot_graph.write_png('Original_Decision_Stump.jpeg')
pydot_graph.set_size('"8,8!"')
pydot_graph.write_png('Resized_Decision_Stump.jpeg')

pydot_graph

print("Execution Completed")


# ### Method 3 : Visualising the Decision Tree using plot_tree() Function

# In[220]:


# Method 3 : Visualising the Decision Stump Using plot_tree() Function

# Displaying the 50th Decision Stump at Index Position 49 in the Base Estimator

# Defining the Decision Stump Graph Data
decision_tree_graph_plot = tree.plot_tree(ada_boost_model.estimators_[49], feature_names=raw_data_pp.drop('Exited', axis=1).columns,
                                         class_names=raw_data_pp['Exited'].unique().astype(str),
                                         filled=True, rounded=True, fontsize=8)

# Visualising the Decision Stump
decision_tree_graph_plot


# In[50]:


# Creating a List of all the Input Features

features = raw_data_pp.drop('Exited', axis=1).columns
feature_names = []
for feature in features:
    feature_names.append(feature)
feature_names


# ### Method 4 : Visualising the Decision Tree in Text Format using export_text() Function

# In[51]:


# Method 4 : Visualising the Decision Stump in Text Format using export_text() Function

# Displaying the 25th Decision Stump at Index Position 24 in the Base Estimator

# Creating a List of Input Feature Names
features = raw_data_pp.drop('Exited', axis=1).columns
feature_names_list = []
for feature in features:
    feature_names_list.append(feature)

# Defining the Decision Stump Textual Representation Data
decision_tree_text = tree.export_text(ada_boost_model.estimators_[24], feature_names=feature_names_list,
                                      spacing=4)

# Visualising the Decision Stump in the Textual Format
print(decision_tree_text)


# ## 9. RETRIEVING THE FEATURE IMPORTANCE VALUES OF THE INPUT FEATURES

# In[52]:


# Retrieving the Information Gain i.e.; Feature Importance Values of the Input Features

# Creating an Empty Data Frame to Hold the Feature Name and the Feature's Importance Values
ig_df_final = pd.DataFrame()

# Looping Through Each and Every Input Feature and Retrieving the Feature Importance Value for Each Feature
for feature, column in enumerate(raw_data_pp.drop('Exited', axis=1)):
    print("{} - {}".format(column, ada_boost_model.feature_importances_[feature]))
    
    # Creating a Data Frame to Include the Feature Name and the Corresponding Feature Importance Value
    ig_df = pd.DataFrame({'Feature': [column], 'Feature Importance': [ada_boost_model.feature_importances_[feature]]})
    
    # Concatenating the Individual Feature Data Frame with the Final Data Frame
    ig_df_final = pd.concat([ig_df_final, ig_df], axis=0, ignore_index=True)
    
# Ordering the Feature Importance Values in the Increasing Order of Importance
ig_df_final_sorted = ig_df_final.sort_values(by='Feature Importance', ascending=False).reset_index(drop=True)
    
ig_df_final_sorted


# ## 10. CALCULATING AND COMPARING THE TRAINING AND VALIDATION ACCURACY

# In[53]:


# Accuracy on the Train Data
print("Training Accuracy: ", ada_boost_model.score(X_train, y_train))

# Accuracy on the Validation Data
print("Validation Accuracy: ", ada_boost_model.score(X_validate, y_validate))


# ## 11. VALIDATING THE CLASSIFIER RESULTS ON THE VALIDATION DATA

# In[54]:


# Validating the Classifier Results on the Validation Data

y_validate_pred = ada_boost_model.predict(X_validate)

y_validate_pred


# ## 12. COMPARING THE VALIDATION ACTUALS WITH THE VALIDATION PREDICTIONS

# In[56]:


# Comparing the Validation Predictions with the Validation Actuals for the first 20 Data Instances

# Validation Actuals
print(y_validate[:20])

# Validation Predictions
print(y_validate_pred[:20])


# ## 13. CONFUSION MATRIX BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[57]:


# Defining the Instance of Confusion Matrix
cm_validation_matrix = confusion_matrix(y_validate, y_validate_pred)

print("Execution Completed")


# ## Method 1 : Plotting the Confusion Matrix with Numeric Values using Seaborn heatmap() Function

# In[58]:


# Method 1 : Plotting the Confusion Matrix with Numeric Values using Seaborn heatmap() Function

churn_cm_plot_1 = sns.heatmap(cm_validation_matrix, annot=True)
churn_cm_plot_1


# ## Method 2 : Plotting the Confusion Matrix with Percentage Values using Seaborn heatmap() Function

# In[59]:


# Method 2 : Plotting the Confusion Matrix with Percentage Values Rounded-off to 2 Decimal Places using Seaborn heatmap() Function

churn_cm_plot_2 = sns.heatmap(cm_validation_matrix/np.sum(cm_validation_matrix), annot=True, fmt='0.2%', cmap='plasma')
churn_cm_plot_2


# ## Method 3 : Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function

# In[60]:


# Method 3 : Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function

cm_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

cm_counts = ["{0:0.0f}".format(value) for value in cm_validation_matrix.flatten()]

cm_percentages = ["{0:0.2%}".format(value) for value in cm_validation_matrix.flatten()/np.sum(cm_validation_matrix)]

cm_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cm_names,cm_counts,cm_percentages)]

cm_labels = np.asarray(cm_labels).reshape(2,2)

sns.heatmap(cm_validation_matrix, annot=cm_labels, fmt='', cmap='jet')


# ## 14. CLASSIFICATION REPORT BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[61]:


# Classification Report and Metrics between the Validation Actuals and the Validation Predictions

target_names = ['No Churn', 'Churn']

# Defining the Classification Report for the Validation Data
classification_report_validation = classification_report(y_validate, y_validate_pred, target_names=target_names)

# Displaying the Classification Report
print(classification_report_validation)


# ## 15. INDIVIDUAL CLASSIFIER METRICS BETWEEN THE VALIDATION ACTUALS AND THE VALIDATION PREDICTIONS

# In[62]:


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


# ## 16. TUNING THE HYPER-PARAMETERS OF THE ADAPTIVE BOOST CLASSIFIER

# In[80]:


# Method

# Creating an Empty Pandas DataFrame to Hold the Tuned Model Results for various combinations of the Hyper-Parameters
ada_boost_tune_model_df_final = pd.DataFrame()

# Setting the Values of the Hyper-Parameters to be used for Tuning the AdaBoostClassifier
base_estimator = DecisionTreeClassifier(max_depth=1, criterion='gini')
n_estimators = [50, 100, 150, 200]                                                     #p1
learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]           #p2
algorithm = ['SAMME', 'SAMME.R']                                                       #p3
random_state = [None, 10]                                                              #p4
      
scenario_id = 0

# Creating a For Loop to Tune the AdaBoostClassifier Model for the Various Combinations of the Hyper-Parameters

for p1, p2, p3, p4 in product(n_estimators, learning_rate, algorithm, random_state):
    
    # Defining the AdaBoostClassifier Model with its Hyper-Parameters
    ada_boost_tune_model = AdaBoostClassifier(base_estimator=base_estimator,
                                             n_estimators=p1,
                                             learning_rate=p2,
                                             algorithm=p3,
                                             random_state=p4)
    
    # Fitting and Training the AdaBoostClassifier Model based on its Hyper-Parameters
    ada_boost_tune_model.fit(X_train, y_train)
    
    # Predicting the Classifier on the Validation Data
    y_validate_tune_pred = ada_boost_tune_model.predict(X_validate)
    
    # Calculating the Accuracy
    accuracy_ada_boost_churn_tune = round((accuracy_score(y_validate, y_validate_tune_pred))*100, 2)
    
    # F1-score
    f1_score_ada_boost_churn_tune = round((f1_score(y_validate, y_validate_pred)*100), 2)

    # Precision
    precision_ada_boost_churn_tune = round((precision_score(y_validate, y_validate_pred)*100), 2)

    # Recall
    recall_ada_boost_churn_tune = round((recall_score(y_validate, y_validate_pred)*100), 2)
    
    # ROC AUC Score
    roc_auc_score_ada_boost_churn_tune = round((roc_auc_score(y_validate, y_validate_pred)*100), 2)
    
    # Incrementing the Scenario_ID for Tracking
    scenario_id += 1
    
    # Displaying the Accuracy Metrics for the Various Combinations of the Hyper-Parameters Tuning
    print(" \n Scenario {} - n_estimators: {}, learning_rate: {}, algorithm: {}, random_state: {}, \n Adaptive Boost Classification Accuracy: {}%, Adaptive Boost Classification F1-Score: {}%, Adaptive Boost Classification Precision: {}%, \n Adaptive Boost Classification Recall: {}%, Adaptive Boost classification ROC AUC Score: {}%".format(scenario_id, p1, p2, p3, p4, accuracy_ada_boost_churn_tune, f1_score_ada_boost_churn_tune, precision_ada_boost_churn_tune, recall_ada_boost_churn_tune, roc_auc_score_ada_boost_churn_tune))
    
    # Defining the Instance of Confusion Matrix
    plot_confusion_matrix(ada_boost_tune_model, X_validate, y_validate)  
    plt.show()
    #title = "Confusion Matrix {} - n_estimators: {}, learning_rate: {}, algorithm: {}, random_state: {}, Adaptive Boost Classification Accuracy: {}%, Adaptive Boost Classification F1-Score: {}%, Adaptive Boost Classification Precision: {}%, Adaptive Boost Classification Recall: {}%, Adaptive Boost classification ROC AUC Score: {}%".format(scenario_id, p1, p2, p3, p4, accuracy_ada_boost_churn_tune, f1_score_ada_boost_churn_tune, precision_ada_boost_churn_tune, recall_ada_boost_churn_tune, roc_auc_score_ada_boost_churn_tune)
    
    # Defining a Individual DataFrame to Hold the Hyper-Parameters Values and their Corresponding Accuracy and ROC AUC Scores
    ada_boost_tune_model_df = pd.DataFrame({'scenario_id': [scenario_id], 'n_estimators': [p1], 'learning_rate': [p2], 'ada boost algorithm': [p3], 'random_state': [p4], 'Accuracy': [accuracy_ada_boost_churn_tune], 'F1-Score': [f1_score_ada_boost_churn_tune], 'Precision': [precision_ada_boost_churn_tune], 'Recall': [recall_ada_boost_churn_tune], 'ROC_AUC_Score': [roc_auc_score_ada_boost_churn_tune]})
    
    # Concatenating the Individual Results DataFrame with the Final DataFrame
    ada_boost_tune_model_df_final = pd.concat([ada_boost_tune_model_df_final, ada_boost_tune_model_df], axis=0, ignore_index=True)
    
# Sorting the Final DataFrame Based on the Increasing Value of Accuracy
ada_boost_tune_model_df_final_sorted = ada_boost_tune_model_df_final.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    
print(ada_boost_tune_model_df_final_sorted)

print("Model Tuning Execution Completed")    


# In[87]:


# Displaying the Various Tuned Model Summarized Results

ada_boost_tune_model_df_final_sorted


# ## 17. VISUALISATION THE PERFORMANCE RESULTS OF THE VARIOUS ADAPTIVE BOOST CLASSIFIER MODELS 

# In[180]:


'''

# Defining the Various Conditions for Visualising the AdaBoostClassifier Model Results

condition_1 = ada_boost_tune_model_df_final_sorted[ada_boost_tune_model_df_final_sorted['n_estimators'] == 50]
condition_2 = condition_1[condition_1['random_state'] != 10]
condition_3 = condition_2[condition_2['ada boost algorithm'] == 'SAMME']

print(condition_3)

# Visualising the AdaBoostClassifier Model Results

print("Number of Decision Stumps = {}".format(n_stumps))
plt.plot(condition_3['learning_rate'], condition_3['Accuracy'])
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy in %")
plt.title("Learning Rate versus Accuracy Curve")
plt.show()

'''


# ### 17.1 Visualisation 1 - Performance Metrics Based on the Total Number of Decision Stumps : random_state = None, ada boost algorithm = SAMME

# In[210]:


# 17.1 Visualisation 1 - Performance Metrics Based on the Total Number of Decision Stumps : random_state = None, ada boost algorithm = SAMME

n_estimators_list = [50, 100, 150, 200]

condition_1 = ada_boost_tune_model_df_final_sorted.loc[(ada_boost_tune_model_df_final_sorted['random_state'] != 10) & (ada_boost_tune_model_df_final_sorted['ada boost algorithm'] == 'SAMME')]

for n_stumps in n_estimators_list:
    result_1 = condition_1[condition_1['n_estimators'] == n_stumps]
    #print(result_1)
    print("Total Number of Decision Stumps = {}".format(n_stumps))
    plt.plot(result_1['learning_rate'], result_1['Accuracy'], label='Accuracy')
    plt.plot(result_1['learning_rate'], result_1['F1-Score'], label='F1-Score')
    plt.plot(result_1['learning_rate'], result_1['Precision'], label='Precision')
    plt.plot(result_1['learning_rate'], result_1['Recall'], label='Recall')
    plt.plot(result_1['learning_rate'], result_1['ROC_AUC_Score'], label='ROC AUC Score')
    plt.xlabel("Learning Rate")
    plt.ylabel("Performance Metrics in %")
    plt.title("Learning Rate versus Performance Metrics in %")
    #plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.figure(figsize=(40, 20))
    plt.show()


# ### 17.2 Visualisation 2 - Performance Metrics Based on the Total Number of Decision Stumps : random_state = 10, ada boost algorithm = SAMME

# In[211]:


# 17.2 Visualisation 2 - Performance Metrics Based on the Total Number of Decision Stumps : random_state = 10, ada boost algorithm = SAMME

n_estimators_list = [50, 100, 150, 200] 

condition_2 = ada_boost_tune_model_df_final_sorted.loc[(ada_boost_tune_model_df_final_sorted['random_state'] == 10) & (ada_boost_tune_model_df_final_sorted['ada boost algorithm'] == 'SAMME')]

for n_stumps in n_estimators_list:
    result_2 = condition_2[condition_2['n_estimators'] == n_stumps]
    #print(result_2)
    print("Total Number of Decision Stumps = {}".format(n_stumps))
    plt.plot(result_2['learning_rate'], result_2['Accuracy'], label='Accuracy')
    plt.plot(result_2['learning_rate'], result_2['F1-Score'], label='F1-Score')
    plt.plot(result_2['learning_rate'], result_2['Precision'], label='Precision')
    plt.plot(result_2['learning_rate'], result_2['Recall'], label='Recall')
    plt.plot(result_2['learning_rate'], result_2['ROC_AUC_Score'], label='ROC AUC Score')
    plt.xlabel("Learning Rate")
    plt.ylabel("Performance Metrics in %")
    plt.title("Learning Rate versus Performance Metrics in %")
    #plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.figure(figsize=(40, 20))
    plt.show()


# ### 17.3 Visualisation 3 - Performance Metrics Based on the Total Number of Decision Stumps : random_state = None, ada boost algorithm = SAMME.R

# In[212]:


# 17.3 Visualisation 3 - Performance Metrics Based on the Total Number of Decision Stumps : random_state = None, ada boost algorithm = SAMME.R

n_estimators_list = [50, 100, 150, 200] 

condition_3 = ada_boost_tune_model_df_final_sorted.loc[(ada_boost_tune_model_df_final_sorted['random_state'] != 10) & (ada_boost_tune_model_df_final_sorted['ada boost algorithm'] == 'SAMME.R')]

for n_stumps in n_estimators_list:
    result_3 = condition_3[condition_3['n_estimators'] == n_stumps]
    #print(result_3)
    print("Total Number of Decision Stumps = {}".format(n_stumps))
    plt.plot(result_3['learning_rate'], result_3['Accuracy'], label='Accuracy')
    plt.plot(result_3['learning_rate'], result_3['F1-Score'], label='F1-Score')
    plt.plot(result_3['learning_rate'], result_3['Precision'], label='Precision')
    plt.plot(result_3['learning_rate'], result_3['Recall'], label='Recall')
    plt.plot(result_3['learning_rate'], result_3['ROC_AUC_Score'], label='ROC AUC Score')
    plt.xlabel("Learning Rate")
    plt.ylabel("Performance Metrics in %")
    plt.title("Learning Rate versus Performance Metrics in %")
    #plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.figure(figsize=(40, 20))
    plt.show()


# ### 17.4 Visualisation 4 - Performance Metrics Based on LBased on the Total Number of Decision Stumps : random_state = 10, ada boost algorithm = SAMME.R

# In[213]:


# 17.4 Visualisation 4 - Performance Metrics Based on the Total Number of Decision Stumps : random_state = 10, ada boost algorithm = SAMME.R

n_estimators_list = [50, 100, 150, 200]

condition_4 = ada_boost_tune_model_df_final_sorted.loc[(ada_boost_tune_model_df_final_sorted['random_state'] == 10) & (ada_boost_tune_model_df_final_sorted['ada boost algorithm'] == 'SAMME.R')]

for n_stumps in n_estimators_list:
    result_4 = condition_4[condition_4['n_estimators'] == n_stumps]
    #print(result_4)
    print("Total Number of Decision Stumps = {}".format(n_stumps))
    plt.plot(result_4['learning_rate'], result_4['Accuracy'], label='Accuracy')
    plt.plot(result_4['learning_rate'], result_4['F1-Score'], label='F1-Score')
    plt.plot(result_4['learning_rate'], result_4['Precision'], label='Precision')
    plt.plot(result_4['learning_rate'], result_4['Recall'], label='Recall')
    plt.plot(result_4['learning_rate'], result_4['ROC_AUC_Score'], label='ROC AUC Score')
    plt.xlabel("Learning Rate")
    plt.ylabel("Performance Metrics in %")
    plt.title("Learning Rate versus Performance Metrics in %")
    #plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.figure(figsize=(40, 20))
    plt.show()


# ### 17.5 Visualisation 5 - Performance Metrics Based on the Learning Rate : random_state = None, ada boost algorithm = SAMME

# In[214]:


# 17.5 Visualisation 5 - Performance Metrics Based on the Learning Rate : random_state = None, ada boost algorithm = SAMME

learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]

condition_5 = ada_boost_tune_model_df_final_sorted.loc[(ada_boost_tune_model_df_final_sorted['random_state'] != 10) & (ada_boost_tune_model_df_final_sorted['ada boost algorithm'] == 'SAMME')]

for lr in learning_rate_list:
    result_5 = condition_5[condition_5['learning_rate'] == lr]
    #print(result_5)
    print("Learning Rate = {}".format(lr))
    plt.plot(result_5['n_estimators'], result_5['Accuracy'], label='Accuracy')
    plt.plot(result_5['n_estimators'], result_5['F1-Score'], label='F1-Score')
    plt.plot(result_5['n_estimators'], result_5['Precision'], label='Precision')
    plt.plot(result_5['n_estimators'], result_5['Recall'], label='Recall')
    plt.plot(result_5['n_estimators'], result_5['ROC_AUC_Score'], label='ROC AUC Score')
    plt.xlabel("Total Number of Decision Stumps")
    plt.ylabel("Performance Metrics in %")
    plt.title("Total Number of Decision Stumps versus Performance Metrics in %")
    #plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.figure(figsize=(40, 20))
    plt.show()


# ### 17.6. Visualisation 6 - Performance Metrics Based on the Learning Rate : random_state = 10, ada boost algorithm = SAMME

# In[215]:


# 17.6. Visualisation 6 - Performance Metrics Based on the Learning Rate : random_state = 10, ada boost algorithm = SAMME

learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]

condition_6 = ada_boost_tune_model_df_final_sorted.loc[(ada_boost_tune_model_df_final_sorted['random_state'] == 10) & (ada_boost_tune_model_df_final_sorted['ada boost algorithm'] == 'SAMME')]

for lr in learning_rate_list:
    result_6 = condition_6[condition_6['learning_rate'] == lr]
    #print(result_5)
    print("Learning Rate = {}".format(lr))
    plt.plot(result_6['n_estimators'], result_6['Accuracy'], label='Accuracy')
    plt.plot(result_6['n_estimators'], result_6['F1-Score'], label='F1-Score')
    plt.plot(result_6['n_estimators'], result_6['Precision'], label='Precision')
    plt.plot(result_6['n_estimators'], result_6['Recall'], label='Recall')
    plt.plot(result_6['n_estimators'], result_6['ROC_AUC_Score'], label='ROC AUC Score')
    plt.xlabel("Total Number of Decision Stumps")
    plt.ylabel("Performance Metrics in %")
    plt.title("Total Number of Decision Stumps versus Performance Metrics in %")
    #plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.figure(figsize=(40, 20))
    plt.show()


# ### 17.7. Visualisation 7 - Performance Metrics Based on the Learning Rate : random_state = None, ada boost algorithm = SAMME.R

# In[216]:


# 17.7. Visualisation 7 - Performance Metrics Based on the Learning Rate : random_state = None, ada boost algorithm = SAMME.R

learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]

condition_7 = ada_boost_tune_model_df_final_sorted.loc[(ada_boost_tune_model_df_final_sorted['random_state'] != 10) & (ada_boost_tune_model_df_final_sorted['ada boost algorithm'] == 'SAMME.R')]

for lr in learning_rate_list:
    result_7 = condition_7[condition_7['learning_rate'] == lr]
    #print(result_5)
    print("Learning Rate = {}".format(lr))
    plt.plot(result_7['n_estimators'], result_7['Accuracy'], label='Accuracy')
    plt.plot(result_7['n_estimators'], result_7['F1-Score'], label='F1-Score')
    plt.plot(result_7['n_estimators'], result_7['Precision'], label='Precision')
    plt.plot(result_7['n_estimators'], result_7['Recall'], label='Recall')
    plt.plot(result_7['n_estimators'], result_7['ROC_AUC_Score'], label='ROC AUC Score')
    plt.xlabel("Total Number of Decision Stumps")
    plt.ylabel("Performance Metrics in %")
    plt.title("Total Number of Decision Stumps versus Performance Metrics in %")
    #plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.figure(figsize=(40, 20))
    plt.show()


# ### 17.8. Visualisation 8 - Performance Metrics Based on the Learning Rate : random_state = 10, ada boost algorithm = SAMME.R

# In[ ]:


# 17.8. Visualisation 8 - Performance Metrics Based on the Learning Rate : random_state = 10, ada boost algorithm = SAMME.R

learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]

condition_8 = ada_boost_tune_model_df_final_sorted.loc[(ada_boost_tune_model_df_final_sorted['random_state'] == 10) & (ada_boost_tune_model_df_final_sorted['ada boost algorithm'] == 'SAMME.R')]

for lr in learning_rate_list:
    result_8 = condition_8[condition_8['learning_rate'] == lr]
    #print(result_5)
    print("Learning Rate = {}".format(lr))
    plt.plot(result_8['n_estimators'], result_8['Accuracy'], label='Accuracy')
    plt.plot(result_8['n_estimators'], result_8['F1-Score'], label='F1-Score')
    plt.plot(result_8['n_estimators'], result_8['Precision'], label='Precision')
    plt.plot(result_8['n_estimators'], result_8['Recall'], label='Recall')
    plt.plot(result_8['n_estimators'], result_8['ROC_AUC_Score'], label='ROC AUC Score')
    plt.xlabel("Total Number of Decision Stumps")
    plt.ylabel("Performance Metrics in %")
    plt.title("Total Number of Decision Stumps versus Performance Metrics in %")
    #plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.figure(figsize=(40, 20))
    plt.show()


# ## 18. CONCLUSION

# ### As we can see from the above results; the better perfroming optimized Ensemble Adaptive Boost Classifier Models are for the Scenarios - 11 & 12 with the below hyper-parameters:
# > ### Total Number of Decision Stumps = 50
# > ### Learning Rate = 0.3
# > ### Ada Boost Algorithm = "SAMME.R"
# > ### Random State = None (or) 10
# 
# ### Performance Metrics and Outcomes for the above mentioned better performing optimized Ensemble Adaptive Boost Classifier Models are:
# > ### Classification Accuracy = 85.5%
# > ### Classification F1-Score = 55.03%
# > ### Classification Precision = 73.23%
# > ### Classification Recall = 44.08%
# > ### Classification ROC AUC Score = 69.88%

# ### 18.1 Re-Training the AdaBoostClassifier Model Identified with the Best Hyper-Parameters Identified Based on the Tuning Process

# In[237]:


# Defining the Parameters of the AdaBoostClassifier Model for Re-Training with the Best Hyper-Parameters Identified Based on the Tuning Process

base_estimator = DecisionTreeClassifier(max_depth=1, criterion='gini')
n_estimators = 50
learning_rate = 0.3
algorithm= 'SAMME.R'
random_state= 10


# Creating an Instance of the AdaBoostClassifier Model with the Best Hyper-Parameters Identified Based on the Tuning Process
ada_boost_best_model_retrain = AdaBoostClassifier(base_estimator=base_estimator,
                                                 n_estimators=n_estimators,
                                                 learning_rate=learning_rate,
                                                 algorithm=algorithm,
                                                 random_state=random_state)

print("Model Training Started.....")

# Re-training the AdaBoostClassifier Model with the Best Hyper-Parameters Identified Based on the Tuning Process
ada_boost_best_model_retrain.fit(X_train, y_train)

print("Model Training Completed.....")


# ### 18.2 Visualizing all the Decision Stumps of the Better Performing Ensemble Adaptive Boost Classifier Model

# In[250]:


# Method 2 : Visualising the Decision Stump Using graph_from_dot_data() Function

# Displaying all the 50 Decision Stumps in the Base Estimator

# Defining the Decision Stump Graph Data

d_stump_counter = 0

for d_stump in range(len(ada_boost_best_model_retrain.estimators_)):
    graph_data_pydot_retrain = tree.export_graphviz(ada_boost_best_model_retrain.estimators_[d_stump], out_file=None, 
                                                    feature_names=raw_data_pp.drop('Exited', axis=1).columns,
                                                    class_names=raw_data_pp['Exited'].unique().astype(str),
                                                    filled=True, rounded=True, special_characters=True,
                                                    impurity=True)

    #graph_data_pydot_retrain

    # Incrementing the Decision Stump Counter by 1
    d_stump_counter += 1

    # Displaying the Decision Stump Count
    #print("\n Adaptive Boost Classifier - Decision Stump {}".format(d_stump_counter))

    # Creating the Decision Stump for the Above Graph Data using pydotplus
    pydot_graph_retrain = pydotplus.graph_from_dot_data(graph_data_pydot_retrain)
    pydot_graph_retrain.set_size('"3,3!"')
    pydot_graph_retrain.write_png('Adaptive_Boost_Classifier_Decision_Stump_{}.jpeg'.format(d_stump_counter))
    
    pydot_graph_retrain
    
print("Execution Completed")


# ![Adaptive_Boost_Classifier_Decision_Stump_1.jpeg](attachment:da312806-103c-4fa9-ba6f-49f4e012c403.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_2.jpeg](attachment:91d5105d-36d1-42d1-be26-1a9f3f401a15.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_3.jpeg](attachment:d8d0738b-ee60-45d9-85b8-25f07bfe9980.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_4.jpeg](attachment:634f650c-8822-4672-b483-1eaf2b7f909b.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_5.jpeg](attachment:8a7da3d2-11c5-4623-8805-9a6407313b10.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_6.jpeg](attachment:df18c5aa-4beb-40bc-a39a-84ae657c791f.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_7.jpeg](attachment:96808532-29e0-41a6-b650-bad3717dea4b.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_8.jpeg](attachment:3c1cde0c-78d0-401f-8f9c-dd6b4d683a50.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_9.jpeg](attachment:87e85014-373e-4c41-b989-440f28521e91.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_10.jpeg](attachment:ad9dc8ef-8293-40d1-9119-3895b40e5c1c.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_11.jpeg](attachment:f0431536-842b-4f33-93e5-d0612fb7862c.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_12.jpeg](attachment:29b50170-e52a-4fbc-b5a9-803065b4d445.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_13.jpeg](attachment:f27d8bb7-d62a-4198-b6c4-9f9b9f19925c.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_14.jpeg](attachment:4504efa2-831e-44cc-94fb-79ace3fa226d.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_15.jpeg](attachment:052fa790-dfe6-4462-9cb8-c79c01806859.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_16.jpeg](attachment:d66341e6-f112-4ecf-8212-7d0c9e2d967e.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_17.jpeg](attachment:66bcfbdd-36ac-4368-96e1-1d57f3bf1333.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_18.jpeg](attachment:f93f1c85-6aa7-4ccd-ac08-9588d57e7ef5.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_19.jpeg](attachment:4162dbb9-ff53-4afa-bc75-14c1cbf1df28.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_20.jpeg](attachment:86069805-ff52-4d6c-ba54-7f3db38a832a.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_21.jpeg](attachment:cc3100d1-9078-4aa0-9179-16ef05bbe33d.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_22.jpeg](attachment:3320206c-5273-49bc-9962-4816c3f7b948.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_23.jpeg](attachment:fa9025cb-3314-41da-87f6-dad8682936e4.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_24.jpeg](attachment:90f97333-3014-4a2b-bf2c-0a278e7fa39f.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_25.jpeg](attachment:86f555ee-687e-431a-932a-322066556411.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_26.jpeg](attachment:97795d73-ada4-4788-b9e3-1f002c78fdde.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_27.jpeg](attachment:a0219e8c-e610-4432-8079-633ea5094647.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_28.jpeg](attachment:c803b916-b9bb-4ac1-8ebf-13542c03445a.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_29.jpeg](attachment:f1425187-8d35-4fc9-80e5-5a338044bb09.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_30.jpeg](attachment:e9d0e21b-f67b-4b2e-a032-9d8f8a54a202.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_31.jpeg](attachment:1ca37a7e-f638-4bfd-a9be-4d0dfb438b80.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_32.jpeg](attachment:77eedc9d-ed80-4698-9438-6917f6cc56c3.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_33.jpeg](attachment:0fe2d2a8-f7fb-46a9-ae7f-e7f694dc7992.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_34.jpeg](attachment:608646c7-dde5-4f56-821e-a9c21cab33a4.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_35.jpeg](attachment:3e3628c6-e1fb-4e3b-b3a2-56020df99551.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_36.jpeg](attachment:017b3946-2d04-42d9-a5ce-698d2761a7db.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_37.jpeg](attachment:70e1fa11-a81d-4081-926e-7f22b8f0550e.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_38.jpeg](attachment:40a2f80b-f3b6-405a-84a9-75b335ec19f3.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_39.jpeg](attachment:8971b8eb-178a-4eb4-aa7e-8455bb6c8e7f.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_40.jpeg](attachment:04e6236d-7d9b-4269-a9ec-747cdf142017.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_41.jpeg](attachment:880e3f14-a79b-475b-a4e6-36187f5aecce.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_42.jpeg](attachment:293f514c-a55e-4ffd-8168-a89e6bfcbcab.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_43.jpeg](attachment:cad7edd7-9b01-4416-af31-4d679c8c1bc2.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_44.jpeg](attachment:695c9aff-b8d2-4847-8042-381536852d0d.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_45.jpeg](attachment:8bbcea63-03bb-44ee-92e8-15037d0fea51.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_46.jpeg](attachment:fdea6d92-7ab7-462f-957a-226abb0105f7.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_47.jpeg](attachment:42050db3-da6b-4855-a832-13173af1accb.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_48.jpeg](attachment:df2586c3-9a6b-4d25-90e2-74327854c578.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_49.jpeg](attachment:91f6c2e7-3d67-49b9-997c-a3d1e1c7832b.jpeg), ![Adaptive_Boost_Classifier_Decision_Stump_50.jpeg](attachment:34cd3d96-b161-45e0-b32a-ba2e69a8e9c4.jpeg)

# ### 18.3 Validating the Classifier Results on the Validation Data for the Better Performing Ensemble Adaptive Boost Classifier Model

# In[254]:


# Validating the Classifier Results on the Validation Data for the Better Performing Ensemble Adaptive Boost Classifier Model

y_validate_pred_best_model = ada_boost_best_model_retrain.predict(X_validate)

y_validate_pred_best_model


# ### 18.4 Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function for the Better Performing Ensemble Adaptive Boost Classifier Model

# In[255]:


# Defining the Instance of Confusion Matrix

cm_validation_matrix_best_model = confusion_matrix(y_validate, y_validate_pred_best_model)

print("Execution Completed")


# In[256]:


# Method 3 : Plotting the Confusion Matrix with Numeric Values, Percentage Values and the Corresponding Text using Seaborn heatmap() Function 
# for the Better Performing Ensemble Adaptive Boost Classifier Model

cm_names_best_model = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

cm_counts_best_model = ["{0:0.0f}".format(value) for value in cm_validation_matrix_best_model.flatten()]

cm_percentages_best_model = ["{0:0.2%}".format(value) for value in cm_validation_matrix_best_model.flatten()/np.sum(cm_validation_matrix_best_model)]

cm_labels_best_model = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cm_names_best_model,cm_counts_best_model,cm_percentages_best_model)]

cm_labels_best_model = np.asarray(cm_labels_best_model).reshape(2,2)

sns.heatmap(cm_validation_matrix, annot=cm_labels_best_model, fmt='', cmap='jet')


# ### 18.5 Classification Report and Metrics between the Validation Actuals and the Validation Predictions for the Better Performing Ensemble Adaptive Boost Classifier Model

# In[257]:


# Classification Report and Metrics between the Validation Actuals and the Validation Predictions 
# for the Better Performing Ensemble Adaptive Boost Classifier Model

target_names_best_model = ['No Churn', 'Churn']

# Defining the Classification Report for the Validation Data
classification_report_validation_best_model = classification_report(y_validate, y_validate_pred_best_model, target_names=target_names_best_model)

# Displaying the Classification Report
print(classification_report_validation_best_model)


# In[ ]:





# In[ ]:




