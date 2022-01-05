#!/usr/bin/env python
# coding: utf-8

# # COMPARISON OF ALL THE CLASSIFIER OPTIMIZED MODELS PERFORMANCE RESULTS AND THE DECISION ON THE BEST PERFORMING FINAL CLASSIFIER MODEL

# ## 1. Decision Tree Classifier - CART (Classification and Regression Tree) Optimized Model

# ![confusion_matrix_decision_tree_classifier_CART.jpg](attachment:e64cf394-ac40-4fa4-8619-a16fc003bc20.jpg)
# ![classification_report_decision_tree_classifier_CART.jpg](attachment:88cfaf43-1f1e-4256-89fa-fb1608cdba67.jpg)

# ## 2. Decision Tree Classifier - IDE (Iterative Dichotomiser) Optimized Model

# ![confusion_matrix_decision_tree_classifier_IDE.jpg](attachment:2ec58f69-0386-4257-a693-eb7d62e8123c.jpg)
# ![classification_report_decision_tree_classifier_IDE.jpg](attachment:d83d5cbf-0816-4bf7-ae84-d207c83ace82.jpg)

# ## 3. Ensemble Random Forest Classifier Optimized Model

# ![confusion_matrix_ensemble_random_forest_classifier.jpg](attachment:4e5cad28-c9db-411a-b610-4c0de101b421.jpg)
# ![classification_report_ensemble_random_forest_classifier.jpg](attachment:881b5505-890a-4466-8713-84f5a8671403.jpg)

# ## 4. Ensemble Adaptive Boosting Classifier Optimized Model

# ![confusion_matrix_ensemble_adaptive_boosting_classifier.jpg](attachment:80b8eb99-0857-444d-b76f-480e8f5039b2.jpg)
# ![classification_report_ensemble_adaptive_boosting_classifier.jpg](attachment:6650dc69-0f47-4dc5-b6f0-95fb5a4afa89.jpg)

# ## 5. Ensemble Hist Gradient Boosting Classifier Optimized Model

# ![performance_metrics_ensemble_hist_gradient_boosting_classifier.jpg](attachment:7c0dccc3-dc5a-4389-bdba-872ece12fe1d.jpg)

# ## 6. Ensemble Extreme Gradient Boosting (XGBoost) Classifier Optimized Model

# ![confusion_matrix_ensemble_extreme_gradient_boosting_classifier.jpg](attachment:eeffa887-4c88-4e01-a008-aa8f47eb71f7.jpg)
# ![classification_report_ensemble_extreme_gradient_boosting_classifier.jpg](attachment:ec02b7cb-e1c3-42a4-9d67-80c5b845d831.jpg)

# ## 7. Support Vector Machine (SVM) Classifier Optimized Model

# ![confusion_matrix_support_vector_machine_classifier.jpg](attachment:e57bd90d-d50f-4cd9-b07b-56f7ba9fb659.jpg)

# ## ALL OPTIMIZED CLASSIFIER MODELS - CONSOLIDATED REPORT ON PERFORMANCE METRICS 

# ![All_Optimized_Models_Comparison_Metrics.jpg](attachment:15134d81-a0ab-4d49-abb9-11495989290d.jpg)

# ## ALL OPTIMIZED CLASSIFIER MODELS - CONSOLIDATED ACCURACY METRICS

# ![All_Optimized_Models_Accuracy_Metrics.jpg](attachment:0c13610d-fa47-469d-bf08-0ed6aeb5dd9c.jpg)

# ### As we can see from the above results; models named "Ensemble Random Forest Classifier Model" and "Ensemble Extreme Gradient Boosting (XGBoost) Classifier Model" have performed comparatively better during the validation stage. 
# 
# ### However; by considering the "Churn" Class Precision Score which is one of the key performance metric in this business case, it is evident that the "Ensemble Extreme Gradient Boosting (XGBoost) Classifier Model" performed significantly much better when compared with that of the "Ensemble Random Forest Classifier Model". 
# 
# ### Hence we can decide and consider "Ensemble Extreme Gradient Boosting (XGBoost) Classifier Model" as the final model to be deployed into the unseen test data.

# In[ ]:




