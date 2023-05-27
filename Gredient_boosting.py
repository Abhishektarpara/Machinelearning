from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score,precision_score ,accuracy_score, recall_score,jaccard_score,make_scorer,balanced_accuracy_score

def Gredient_boosting(X, Y):
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.05, max_features=2, max_depth=2,
                                        random_state=0)
    #applying KFold cross validation for accuracy score,f1 score,recall,precision,jaccard
    _scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro',zero_division='warn'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        'jaccard': make_scorer(jaccard_score, average='macro'),
    }
    results = cross_validate(
        estimator=gb_clf,
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