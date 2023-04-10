## US Primaries Election Analysis

This repository contains an analysis of the 2016 United States Primaries Election, focusing on predicting whether Donald Trump received more than 50% of the votes and clustering counties based on their demographic characteristics. The analysis includes the use of five distinct methods and eight distinct models were created for predictive modeling and an explanation of the clustering methodology through the economics variables related to the voters.

To be more precise, in this analysis, we worked on two parts of a problem related to the 2016 US Presidential Election. In the first part, we developed a predictive model to classify whether Donald Trump received more than 50% of the votes in the Republican primaries. We employed various classification techniques such as **Logistic Regression, Decision Tree, Naive Bayes, Support Vector Machines, and K-Nearest Neighbors (four distinct models)**. We evaluated the models using various statistical measures, including Accuracy, Precision, Recall, Lift and ROC Curves, AUC Values and Brier Score, and compared their performance. In the second part, we applied clustering techniques to group the counties based on demographic variables and used economic variables to describe the clusters. We used **Hierarchical and K-Means clustering algorithms** and performed ANOVA-based variable selection to improve clustering analysis. Finally, we profiled the clusters based on their characteristics.

---

### Introduction - Description of the Problem
The problem consists of two parts. The first part is to develop a predictive model that classifies whether Donald Trump received more than 50% of the votes in the US Presidential Election based on the 2016 primary election results from the Democratic and Republican parties. The second part is to cluster the counties based on their demographic characteristics using certain variables, and then describe the clusters using economic variables. The report will utilize at least three distinct methods for the predictive model and there are no constraints on the clustering methodology. The objective is to provide insights into the effectiveness of various modeling techniques and their ability to predict electoral outcomes accurately.

---

### Data Cleaning and Data Transformation

In the Data Cleaning and Transformation stage of the study, the 'votes' dataset was analyzed to construct a model that can explain the behavior of voters in counties where Donald Trump received more than 50% of the Republican votes. To achieve this, the dataset was filtered to exclude observations related to the Democratic party and candidates. The 'votes' dataset was found to contain 20 Federal Information Processing Standards (FIPS) county codes missing for the 4 candidates in the New Hampshire (NH) state, which were selectively imported by accurately identifying the New Hampshire counties. Some states have a distinct but unknown method of assigning codes to their counties, utilizing an 8-digit encoding format instead of the standard 5-digit FIPS codes. Alternative approaches were recognized for these states, and four separate CSV files were assembled through web-based research that contained the relationships between counties and cities/towns to assign the counties to the appropriate FIPS codes using R and the 'county_facts' and 'votes' datasets. However, for the states of Alaska, Kansas, and Wyoming, it was not possible to accurately assign the votes to individual counties due to the nature of the provided data collection.

--- 

### Data Partitioning

In data partitioning, we divide a dataset into two or more subsets to train and evaluate the model's performance. The purpose of this process is to avoid overfitting and maintain generalization, which occurs when the model is too complex and fits the training data too closely, resulting in poor performance on new data. To ensure a standardized and robust way to train and evaluate models, we use the trainControl function, which implements repeated K-fold cross-validation. This method divides the data into K-folds, trains and evaluates the model k times, and averages the results to obtain a more reliable estimate of the model's performance. We also set the random seed to ensure reproducibility of the analysis. By partitioning the data and using cross-validation, we can train and evaluate models in a consistent, robust, and reliable way.

--- 

### Part I: Classification
---

Classification is a fundamental task in machine learning, which involves assigning a label or a category
to a given input or observation. The goal of classification is to build a model that can accurately predict
the label of new, unseen data based on patterns and relationships identified in the training data.

Classification has many practical applications in various fields, such as image recognition, spam
filtering, credit scoring, medical diagnosis, and sentiment analysis.
To build a classification model, we typically start with a labeled dataset, where each data point is
associated with a class label. We use this data to train a model that can learn the underlying patterns
and relationships between the input features and the output labels. Once the model is trained, we can
use it to predict the labels of new, unseen data and evaluate its performance through metrics such as
accuracy, precision, recall, and ROC curve among others.

---

###  Logistic Regression
Logistic regression is a statistical method used for modeling the relationship between a binary dependent variable and one or more independent variables. It is a type of generalized linear model that uses a logistic function to model the probability of the dependent variable being in one of the two categories, based on the values of the independent variables. Logistic regression is widely used in various fields, such as medicine, epidemiology, social sciences, marketing, and finance, for analyzing binary outcomes and making predictions about future events.

To build a logistic regression model, we typically start with a labeled dataset, where each data point is associated with a binary class label. We use this data to train a model that can learn the underlying patterns and relationships between the input features and the output labels. The logistic regression model estimates the coefficients of the independent variables, which indicate the direction and magnitude of their effects on the log odds of the positive class. These coefficients can be exponentiated to obtain odds ratios, which represent the multiplicative change in the odds of the positive class for a one-unit increase in the corresponding independent variable.

To select the best subset of variables for the logistic regression model, we can use a stepwise selection procedure based on the Akaike Information Criterion (AIC). We can also check for multicollinearity issues among the independent variables in the model using the Variance Inflation Factor (VIF) values. Multicollinearity occurs when two or more independent variables in a logistic regression model are highly correlated, which can cause them to share variation and make it difficult for the model to distinguish their unique effects on the dependent variable. By addressing multicollinearity, we can improve the accuracy and reliability of the predictive model.

Once the logistic regression model is built, we can evaluate its performance through metrics such as the Wald test, Likelihood Ratio Test, and pseudo-R2 (McFadden, CoxSnell, and Nagelkerke). The Wald test measures whether at least one of the coefficients in the model is significantly different from zero. The Likelihood Ratio Test (LRT) compares the full model to the null model (which only includes the intercept) to see if the full model is a significantly better fit. The pseudo-R2 measures how much of the variation in the dependent variable is explained by the independent variables in the model.

---
