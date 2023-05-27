# Machinelearning
Here I have taken 5 datasets and applied 5 diffrent Classification algorithm on those datasets and comparision of accuracy.


# Classification Algorithms

Classification and class probability estimation attempts to 
predict , for each individual in a population, to which class does 
this individual belongs to. Generally the classes are independent of 
each other.



## Classifiers
KNN-Classifier, Decision tree, Gredient boosting, Naive bayes,
Random forest Classifier
## KNN-Classifier
The K-Nearest Neighbour or the KNN algorithm is a machine learning algorithm based on the supervised learning model. The K-NN algorithm works by assuming that similar things exist close to each other. Hence, the K-NN algorithm utilises feature similarity between the new data points and the points in the training set (available cases) to predict the values of the new data points. In essence, the K-NN algorithm assigns a value to the latest data point based on how closely it resembles the points in the training set. K-NN algorithm finds application in both classification and regression problems but is mainly used for classification problems.
## Decision_tree
A decision tree is one of the popular as well as powerful tools which is used for prediction and classification of the data or an event. It is like a flowchart but having a structure of a tree. The internal nodes of the trees represent a test or a question on an attribute; each branch is the possible outcome of the question asked, and the terminal node, which is also called as the leaf node, denotes a class label. 
## Gredient-boosting

Gradient boosting is a technique used in creating models for prediction. The technique is mostly used in regression and classification procedures. Prediction models are often presented as decision trees for choosing the best prediction. Gradient boosting presents model building in stages, just like other boosting methods, while allowing the generalization and optimization of differentiable loss functions.
## Random-forest
Assuming your dataset has “m” features, the random forest will randomly choose “k” features where k < m.  Now, the algorithm will calculate the root node among the k features by picking a node that has the highest information gain. 
After that, the algorithm splits the node into child nodes and repeats this process “n” times. Now you have a forest with n trees. Finally, you’ll perform bootstrapping, ie, combine the results of all the decision trees present in your forest.
It’s certainly one of the most sophisticated algorithms as it builds on the functionality of decision trees. 
## Naive-bayes
A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem.


    Naïve: It is called Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of color, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other.
    Bayes: It is called Bayes because it depends on the principle of Bayes' Theorem.

## Comparing Scores
Here we are comparing 5 diffrent accuracy scores(F1,jaccard,Recall,Precision,accuracy) on 5 diffrent dataset with KFold cross validation.
## Datasets-used
datasets are given inform of tables.

    1.Customer.csv,
    2.Gender_classification.csv,
    3.house_votes.csv,
    4.Movie_classification,
    5.Wine.csv
## Run
For running this project in cmd give following command.
it will take datasets from datasets folder, and
it will automatically create new output file called output.xlsx

    python main.py
## Implementation
Train model based on given training dataset and after using trained
model we try to predict values for testing data.
and for verification we also use diffrent accuracy measuring t
techniques. 

Implement all classifires using scikit-learn in-built library.

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB

Input: Given dataset in one folder and for all datasets apply 
classifires, input form is 2D matrix.

verification and Accuracy: For finding how well our model trained
on given dataset use some Accuracy scores.

    from sklearn.metrics import f1_score,precision_score ,accuracy_score, recall_score,jaccard_score,make_scorer







## F1-score
F1-Score or F-measure is an evaluation metric for a classification defined as the harmonic mean of precision and recall. It is a statistical measure of the accuracy of a test or model.
## Jaccard-score
Jaccard Similarity is a common proximity measurement used to compute the similarity between two objects, such as two text documents. Jaccard similarity can be used to find the similarity between two asymmetric binary vectors or to find the similarity between two sets. In literature, Jaccard similarity, symbolized by , can also be referred to as Jaccard Index, Jaccard Coefficient, Jaccard Dissimilarity, and Jaccard Distance.
## Recall
The recall is none other than the ratio of True Positive and the sum of True Positive and False Negative.nstead of looking at False Positive values Recall looks for False Negative values. Recall value depends on the False Negative.
## Precision
Precision is no more than the ratio of True Positive and the sum of True Positive and False Positive.
If the ratio of Precision is 50%, then the predicted output values of our model are 50% correct.
## k-Fold Cross-Validation

The general procedure is as follows:

    Shuffle the dataset randomly.
    Split the dataset into k groups
    For each unique group:
        1.Take the group as a hold out or test data set
        2.Take the remaining groups as a training data set
        3.Fit a model on the training set and evaluate it on the test set
        4.Retain the evaluation score and discard the model
    Summarize the skill of the model using the sample of model evaluation scores

## Output
Run main.py file and output will be stored in output.xlsx file.All Classication techniques shown are applied on all datasets 
and results of all Accuracy are stored on xlsx files.
    
    main.py
    output.xlsx
