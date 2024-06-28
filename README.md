# Predicting Marijuana Use Disorder: A Machine Learning Approach

### Overview

The goal of this project is to investigate the socio-demographic factors potentially associated with the risk of a marijuana user developing substance use disorder using a machine learning approach. By leveraging advanced machine learning models, I aim to identify key predictors and improve our understanding of the factors contributing to marijuana use disorder. The methodology I used is based on Rajapaksha et al. (2020), in which the researchers employed LASSO, KNN, Random Forest, SVM, and Gradient Boosting to estimate the chance of developing SUD based on various demographic, behavioral, psychiatric, and cognitive risk factors. 

In this project, I will use logistic regression, Lasso logistic regression, Random Forest, and Gradient Boosting The overall performance of the machine learning models will be evaluated using the area under the receiver operating characteristic curve (AUC), overall accuracy (i.e., the proportion of overall correct classifications), sensitivity (i.e., the proportion of correct classifications among the SUD instances/true positives), and specificity (i.e., the proportion of correct classifications among the non-SUD instances/true negatives). By comparing these metrics, I aim to determine the most effective machine learning model for predicting the risk of developing marijuana use disorder. This project can inform prevention and intervention strategies, ultimately aiding in addressing the challenges posed by increased marijuana use.

### Data

[2022 National Survey on Drug Use and Health (NSDUH) Releases](https://www.samhsa.gov/data/release/2022-national-survey-drug-use-and-health-nsduh-releases)

[DSM-5 Diagnostic Criteria for Diagnosing and Classifying Substance Use Disorders](https://www.ncbi.nlm.nih.gov/books/NBK565474/table/nycgsubuse.tab9/)

### Project Roadmap

1. **Project Setup and Data Acquisition**
   - [x] Create project repository
   - [x] Acquire NSDUH 2022 dataset
   - [x] Install necessary packages and libraries

2. **Data Cleaning and Preprocessing**
   - [x] Load dataset and handle missing values
   - [x] Reorganized some variables into more general levels
   - [x] Split dataset into training and testing sets

3. **Exploratory Data Analysis (EDA)**
   - [x] Visualize data distributions
   - [x] Identify correlations between predictors and outcome variable

4. **Feature Engineering**
   - [x] Create new features if necessary
   - [x] Perform one-hot encoding for categorical variables
   - [x] Select relevant features based on EDA and domain knowledge

5. **Model Training**
   - [x] Train Logistic Regression model
   - [x] Train Lasso Logistic Regression model
   - [x] Train Random Forest model
   - [x] Train Gradient Boosting model
   - [x] Implement cross-validation for each model

6. **Model Evaluation**
   - [x] Evaluate models using accuracy, precision, recall, F1-score, and AUC
   - [x] Compare model performance
   - [x] Select the best-performing model

7. **Interpretation and Insights**
   - [x] Interpret the coefficients of the best-performing model
   - [x] Analyze feature importance
   - [x] Draw conclusions from the model outputs

8. **Reporting**
   - [x] Create a comprehensive report of findings
   - [x] Document methodology and results
   - [x] Prepare visualizations for key insights

9. **Future Work**
    - [ ] Explore additional models and techniques
    - [ ] Incorporate more recent data
    - [ ] Improve model performance with advanced feature engineering





