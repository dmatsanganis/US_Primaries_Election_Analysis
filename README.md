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

## Part I: Classification

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

### Decision Tree

Decision Trees are a popular machine learning technique used for solving classification and regression
problems. They provide a visual and intuitive representation of the decision-making process by
recursively partitioning the data based on the most relevant features

The model is trained using the training dataset, and the testing dataset is used to evaluate model performance. The optimal tree size is determined by printing the complexity parameter table, which outputs the rules used to build the decision tree model. The root node error is calculated, and the relative error and cross-validation error rates associated with each split in the decision tree are shown. The importance of each independent variable in the model is also output. The model is then used to predict the outcomes of the test data, and a confusion matrix is created to evaluate the accuracy of the model. The decision tree model metrics achieved an overall accuracy of 0.8744, with a 95% confidence interval ranging from 0.8499 to 0.8962. The Kappa score for the model was 0.7379, and the sensitivity of the model is 0.8711, while the specificity of the model is 0.8765. The positive predictive value (PPV) of the model is 0.8147, and the negative predictive value (NPV) of the model is 0.916. The text concludes that the Decision Tree model showed a promising performance in predicting binary outcomes, achieving a high level of accuracy and substantial agreement with the observed outcomes, but there is still room for improvement in reducing the number of false negatives and false positives.

Furthermore, since the decision tree algorithm recursively splits the dataset into smaller and smaller subsets based on the values of the input features, with the goal of creating a model that can accurately predict the target variable. In our case, the target variable is the predicted percentage of votes for Donald Trump in the 2016 U.S. presidential election. The decision tree model uses a set of rules to predict the target variable based on the values of the input features for each observation. The rules are based on the relationships between the input features and the target variable that were learned from the training data. The decision tree model is useful because it provides interpretable results that can be easily understood by non-experts, making it a valuable tool for decision-making in our analysis.

---

### Naive Bayes

The Naive Bayes model is a probabilistic algorithm that works on the principle of Bayes'
theorem, which calculates the probability of an event occurring based on prior knowledge of related
events. Furthermore, Naive Bayes uses Bayes' theorem to predict the probability of a particular data
point belonging to a certain class. It is a simple and efficient algorithm that works well in a variety of
settings, including natural language processing, spam filtering, and image recognition. The "naive" in
its name comes from the assumption that all features in the data are independent of each other,
which is often not true in real-world data (and in our case too). Despite this limitation, Naive Bayes
remains a popular and effective algorithm due to its simplicity and speed.

The Naive Bayes model is constructed using the naiveBayes function from the e1071 library. The response variable is set to whether Donald Trump got the majority of Republicans votes in each county, while all other variables in the dataset are used as predictors. The model is trained on a training set that was previously created using data partitioning techniques to ensure robustness and accuracy.
The performance of the Naive Bayes model is evaluated using a confusion matrix, which shows the number of correct and incorrect predictions for each class. The accuracy, Kappa statistic, sensitivity, specificity, positive predictive value, and negative predictive value are also calculated. The results indicate that the model performed well on the dataset, with high accuracy, Kappa statistic, and sensitivity/specificity values.
Finally, an ROC curve is outputted for the model, which shows the tradeoff between the true positive rate and the false positive rate as the decision threshold for the classifier is varied.

**However**, it is important to note that although the Naive Bayes model performed well on the given dataset, there are several limitations and assumptions associated with this algorithm. For instance, the independence assumption is violated in real-world scenarios, and the accuracy of the model can be affected as a result. Additionally, Naive Bayes requires a relatively large amount of data to build an accurate model, and in this case, the train dataset only had 1935 observations. Another limitation is that Naive Bayes is very sensitive to outliers and the normal assumption, which may not hold true for some variables in the dataset. Therefore, when comparing the performance of different models, we should be cautious and take these limitations and assumptions into consideration.

---

### Support Vector Machine (SVM) 

Support Vector Machine (SVM) is a supervised machine learning algorithm used for
classification and regression tasks. It works by finding the best hyperplane in a high-dimensional space
that separates different classes or groups of data points. The hyperplane that maximizes the margin
between the classes is chosen as the optimal one. SVM has been widely used in various applications
such as image classification, text classification, and bioinformatics.

The procedure followed in this section involves using Support Vector Machines (SVM) as a supervised machine learning algorithm for classification. 
The dataset is first checked for dimensions and then scaled to ensure all predictors are on the same scale. The SVM model is trained using the svmLinear 
method to classify data into two classes based on the response variable "trump_over_perc50". The training data is used to train the model, while the unseen 
data is used to test the accuracy of the model. The trained SVM model is then used to predict the class labels of the test data set, and a confusion matrix 
is created to compare the predicted class labels with the actual class labels of the test data set. The overall accuracy, Kappa statistic, specificity, 
sensitivity, positive predictive value (PPV), negative predictive value (NPV), prevalence of the positive class label, and detection rate are calculated 
to evaluate the performance of the model. Finally, the ROC curve is plotted to visualize the performance of the binary classifier model.

---

### K-Nearest Neighbors (KNN) 

K-Nearest Neighbors (KNN) algorithm, a popular machine learning algorithm used for classification tasks. KNN is an instance-based learning algorithm that memorizes the training dataset and works by calculating the distance between the input data point and all the other data points in the training set. It then selects the K-nearest data points based on the calculated distance, and the classification of the input data point is determined by the majority class of the K-nearest neighbors. To build the KNN models we follow the next steps. First, we check the dimensions of the training and testing datasets to ensure they are equal. Then, 
we scale the data to ensure that all the features are given equal weight in the distance metric. Thirdly, we define a grid of K values ranging from 1 to 20 and using 10-fold cross-validation to evaluate the performance of the KNN algorithm for each value of K. Afterwards, we fit the KNN model with different values of K and tuning the hyperparameters using the train function from the caret library. Lastly, we evaluate the performance of the KNN algorithm using confusion matrix, classification accuracy, no information rate (NIR), kappa statistic, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), detection rate, detection prevalence, and ROC curves.

According to the analysis, the results showed that the optimal number of nearest neighbors for the KNN algorithm is 5, closely followed by 4 NN. Therefore, we performed KNN classification with both scaled and unscaled datasets using 5 and 4 NN, and evaluated the performance of the KNN algorithm. To sum up, we built a total of 4 KNN models (scaled with 4 NN, unscaled with 4 NN, scaled with 5 NN, and unscaled with 5 NN) and evaluated their performance using the aforementioned metrics.

---

### Models Comparisons

Model comparison is an important step in machine learning and involves evaluating the performance
of different models and selecting the best one for the task at hand. To be more precise, we will
present various measures to select the best of the aforementioned models.

The given report discusses the process of comparing different machine learning models to select the best one for a particular task.
The report first discusses the importance of checking the balance of the dataframes before proceeding with model comparisons. 
It then suggests using multiple performance metrics like sensitivity, specificity, and precision in addition to accuracy to evaluate model performance in imbalanced data scenarios.

Next, the report's procedure excludes the KNN models with four and five nearest neighbors due to poor results and suggests comparing the remaining six models - Logistic Regression,
Decision Tree, SVM, Naïve Bayes, KNN with K=5 scaled, and KNN with K=4 scaled.

Finally, the report presents a table showing statistics measures like accuracy, precision, recall/sensitivity, specificity, and F1 score for all six models. 
It suggests that the Naïve Bayes model performs the best on the given dataset with the highest accuracy and precision scores, followed by Logistic Regression and SVM. 
The F1 score is also discussed as a useful metric when the classes are imbalanced, as it measures the harmonic mean of precision and recall.

---

### Conclusion

After taking all the aforementioned measures into consideration, and by excluding the Naïve Bayes
models due to certain assumptions and limitations, such as assuming that the features are
independent and that all features are equally important, we reach to the conclusion that the Logistic
Regression and the SVM model are the leading two followed by the Decision Tree model. However
since we are aiming for the best predictive model - and we do not care about the interpretations - the
Logistic Regression model performs slightly better than the SVM and the Decision Tree models across
to the multiple measures that were implemented above. Therefore, our final choice would be the
**Logistic Regression** model.

---

## Part II: Clustering

In the context of unsupervised learning, we will be exploring the use of clustering methods to extract
valuable insights from our dataset. In contrast to supervised learning, the dataset lacks any labeled
information, thus our objective is to discover hidden structures or groupings among the data points.

Our aim is to partition the data into distinct clusters, where each cluster comprises of observations
that exhibit similar characteristics or attributes. The clustering process is motivated by the goal of
identifying homogeneity within clusters and heterogeneity between them, without any prior
knowledge of the underlying labels or classes. Clustering is used in various domains such as marketing,
image segmentation, bioinformatics, and social network analysis, among others.

Clustering algorithms can be broadly categorized into two types: hierarchical clustering and
partitioning clustering. In hierarchical clustering, the data points are recursively grouped into a treelike structure,
known as a dendrogram, based on the similarity between them. The dendrogram is
then pruned to obtain the final clusters. On the other hand, in partitioning clustering, the data points
are divided into a fixed number of clusters based on a predefined criterion, such as minimizing the
distance between the data points in the same cluster. The two most popular clustering algorithms –
which will also be applied in the current analysis are the K-Means and the Hierarchical Clustering.













---

### Datasets Archive

All the datasets created and/or used during this analysis (and the ones to relate the county_facts with the votes
Excel sheets), as well as the provided one, can be found on the Datasets Archive file. These are the followings:

▪ The given dataset that contains three Excel sheets, stat BA II project I.xlsx

▪ Massachusetts (MA) assignment of county to town/city CSV file, MA.csv

▪ Connecticut (CT) assignment of county to town/city CSV file, CT.csv

▪ Rhode Island (RI) assignment of county to town/city CSV file, RI.csv

▪ Vermont (VT) assignment of county to town/city CSV file, VT.csv

▪ The votes dataset with the proper FIPS codes, filtered only for the Republicans votes, and
containing only the response variable CSV file, us_elections_2016.csv

▪ The latest dataset after the merge with the 'county_facts' Excel sheet of the stat BA II project I.xlsx
dataset. This is the final dataset of the data cleaning phase, us_elections_2016FINAL.csv

▪ The clustering dataset based on the stat BA II project I.xlsx without the response variable, the
states and the fips codes but with the addition of the county variable (data points based on
counties), us_elections_2016_clustering.csv

You can also find an interactive clustering explanation based on the economics related variables here:

▪ Economics explanation/profiling clusters plot (Economics_explanation_clusters.html).




