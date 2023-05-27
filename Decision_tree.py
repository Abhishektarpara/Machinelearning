from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score,precision_score ,accuracy_score, recall_score,jaccard_score,make_scorer

def Decision_tree(X, Y):
    classifier = DecisionTreeClassifier()
    #applying KFold cross validation for accuracy score,f1 score,recall,precision,jaccard
    _scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        'jaccard': make_scorer(jaccard_score, average='macro')
    }
    results = cross_validate(
        estimator=classifier,
        X=X,
        y=Y,
        cv=5,
        scoring=_scoring,
        return_train_score=True
    )
    return {"Mean Validation Accuracy score": results['test_accuracy'].mean() * 100,
            "Mean Validation Precision score": results['test_precision'].mean() * 100,
            "Mean Validation Recall score": results['test_recall'].mean() * 100,
            "Mean Validation F1 Score": results['test_f1'].mean() * 100,
            "Mean validation Jaccard Score": results['test_jaccard'].mean() * 100}