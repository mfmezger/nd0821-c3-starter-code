# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
An AdaBoost [1] classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier

## Intended Use
This AdaBoost classifier is intended to be used on census data to predict a specific target variable the income level . The model can be used to identify patterns and relationships between different demographic and socioeconomic characteristics, which can be helpful in informing policy decisions, resource allocation, and other applications.

## Training Data
The training data for this model is obtained from the UCI Machine Learning Repository's Census Income Dataset (also known as the Adult Dataset). The dataset contains around 32k instances and 14 attributes, including age, work class, education, marital status, occupation, and income. The data has been preprocessed and cleaned to remove instances with missing values and outliers. The dataset is split into 80% training and 20% testing data.

## Evaluation Data
The evaluation data is the remaining 20% of the UCI Machine Learning Repository's Census Income Dataset that was not used for training. This dataset contains 9,768 instances and the same 14 attributes as the training data. The evaluation data is used to measure the model's performance and generalization ability.

## Metrics
The model's performance is evaluated using the following metrics:

Accuracy: The proportion of correctly classified instances out of the total instances.
Precision: The proportion of true positive instances out of the instances predicted as positive.
Fbeta: A weighted harmonic mean of precision and recall, where recall is the proportion of true positive instances out of the total positive instances.

precision: 0.7557312252964427, recall: 0.62853385930309, fbeta: 0.6862885857860731

## Ethical Considerations
When using this model, it is crucial to consider the potential biases present in the training data, such as sampling bias or underrepresentation of certain demographic groups. These biases could lead to the model producing unfair or discriminatory predictions. It is essential to evaluate the model's performance and fairness across different subgroups and to take appropriate measures in addressing any identified disparities.

## Caveats and Recommendations
The model's performance may be sensitive to the choice of base estimator and the number of boosting iterations. It is recommended to perform a grid search or other hyperparameter optimization techniques to find the best model configuration for your specific use case.
AdaBoost is susceptible to overfitting when the base estimator is too complex or the data is noisy. Be cautious when interpreting the model's predictions in these cases.
The model has been trained on a specific census dataset and may not generalize well to other datasets with different distributions or characteristics. It is advised to validate the model on relevant data before using it for decision-making purposes.