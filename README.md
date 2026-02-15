# 📘 Phishing Website Detection Using Machine Learning

## Machine Learning – Assignment 2  
**M.Tech (AIML / DSE), BITS Pilani**

---

## a. Problem Statement

Phishing websites are malicious web pages created with the intent of misleading users into disclosing confidential information such as usernames, passwords, and financial details. As online transactions and digital services continue to grow, phishing attacks pose a significant threat to cybersecurity.

The aim of this project is to design, implement, and evaluate multiple machine learning–based classification models that can automatically identify phishing websites. Numerical features extracted from URLs are used for classification, and an interactive Streamlit application is developed to visualize model performance and evaluation results.

---

## b. Dataset Description

- **Problem Type:** Binary Classification  
- **Total Records:** Approximately 11,000  
- **Number of Features:** 87 numerical attributes  
- **Target Variable:** `status`  
  - `legitimate` → 0  
  - `phishing` → 1  

The dataset contains engineered numerical features derived from website URLs and related properties, such as URL length, use of HTTPS, special characters, and structural patterns. During preprocessing, the raw URL column is excluded, and only numerical features are retained for model training. The dataset is complete and does not contain missing values.

---

## c. Machine Learning Models and Evaluation Metrics

All classification models were trained and tested on the same dataset using a **stratified 80–20 train–test split**, ensuring consistency and fairness in performance comparison across models.

### Implemented Classification Models

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

### Evaluation Metrics

To assess model performance, the following metrics were computed for each classifier:

- Accuracy  
- Area Under the ROC Curve (AUC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

### Model Comparison Summary

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9239 | 0.9770 | 0.9239 | 0.9239 | 0.9239 | 0.8478 |
| Decision Tree | 0.9348 | 0.9348 | 0.9349 | 0.9348 | 0.9348 | 0.8697 |
| KNN | 0.9361 | 0.9685 | 0.9362 | 0.9361 | 0.9361 | 0.8724 |
| Naive Bayes | 0.7069 | 0.9274 | 0.7863 | 0.7069 | 0.6851 | 0.4868 |
| Random Forest | 0.9615 | 0.9936 | 0.9615 | 0.9615 | 0.9615 | 0.9230 |
| XGBoost | 0.9672 | 0.9944 | 0.9672 | 0.9672 | 0.9672 | 0.9344 |

---

### Model-wise Performance Observations

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Achieves 92.39% accuracy with a high AUC of 0.9770, indicating strong overall discrimination. Performance is stable but slightly limited by the linear decision boundary when features have non-linear interactions. |
| Decision Tree | Reaches 93.48% accuracy and captures non-linear patterns well. However, the AUC (0.9348) equals the accuracy, suggesting the probability estimates are not well-calibrated, which is typical for unpruned decision trees. |
| KNN | Attains 93.61% accuracy with a solid AUC of 0.9685. Distance-weighted voting helps improve predictions, though inference cost scales with dataset size. |
| Naive Bayes | Records the lowest accuracy at 70.69% and a weak MCC of 0.4868, indicating that the strong feature independence assumption does not hold well for this dataset. Precision (0.7863) is notably higher than recall (0.7069), showing many phishing sites are missed. |
| Random Forest | Delivers strong performance with 96.15% accuracy and an AUC of 0.9936. The ensemble of 150 trees effectively reduces variance and captures complex feature interactions. |
| XGBoost | Achieves the best overall performance with 96.72% accuracy and the highest AUC of 0.9944. Gradient boosting with 200 estimators excels at minimizing both bias and variance, making it the most reliable model for phishing detection on this dataset. |

---

## Streamlit Web Application

An interactive web application was developed using **Streamlit** to demonstrate and compare the implemented machine learning models. The application supports the following functionalities:

- Uploading external test datasets in CSV format  
- Selecting classification models through a dropdown interface  
- Displaying evaluation metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and MCC  
- Visualizing the confusion matrix  
- Presenting a structured classification report  

---

## Repository Structure

```
ML_Assignment_2/
│── app.py
│── requirements.txt
│── README.md
│
├── data/
│   └── dataset_phishing.csv
│
└── model/
    │── preprocess.py
    │── metrics.py
    │── logistic_regression_model.py
    │── decision_tree_model.py
    │── knn_classifier_model.py
    │── naive_bayes_model.py
    │── random_forest_model.py
    └── xgboost_model.py
```

---

## How to Run the Project Locally

1. Create and activate a Python virtual environment  
2. Install all required dependencies using `requirements.txt`  
3. Launch the Streamlit application using:  
   ```bash
   streamlit run app.py
   ```

---

## Deployment

The application has been deployed using **Streamlit Community Cloud**. A clickable link to the live application is included in the final PDF submission, as required by the assignment guidelines.

---

## Implementation Note

This project follows a modular machine learning workflow in which data preprocessing, model training, evaluation, and user interface components are implemented as separate and reusable units. Although the same dataset and model categories were used as specified in the assignment requirements, the overall pipeline structure, code organization, and Streamlit integration were independently designed to ensure clarity, reproducibility, and maintainability.

---

## Conclusion

This project presents a complete end-to-end machine learning workflow encompassing data preprocessing, model development, performance evaluation, comparative analysis, and web-based deployment. Ensemble-based classifiers, particularly Random Forest and XGBoost, demonstrate superior performance, highlighting their suitability for phishing website detection tasks.

---

## Author

**Name:** Kamakshi Jayanthi  
**Program:** M.Tech (AIML / DSE)  
**Institute:** BITS Pilani  
