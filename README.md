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
