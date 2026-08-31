"""
Model Definition Module (Point 6)
=================================
Initializes the exact 7 core classifier algorithms with fixed random seeds (42).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_core_models():
    """
    Returns dictionary of the exact 7 classifiers.
    """
    models = {
        'Logistic Regression': (
            LogisticRegression(C=10.0, solver='lbfgs', max_iter=4000, random_state=42),
            True,
            'Standard linear logistic logit classifier'
        ),
        'Linear Discriminant Analysis (LDA)': (
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            True,
            'Linear discriminant boundary'
        ),
        'Support Vector Machine (LinearSVC)': (
            LinearSVC(C=5.0, loss='squared_hinge', random_state=42, max_iter=4000),
            True,
            'Linear support vector machine separator'
        ),
        'Gaussian Naive Bayes': (
            GaussianNB(var_smoothing=1e-2),
            True,
            'Probabilistic naive bayes baseline'
        ),
        'Decision Tree Classifier': (
            DecisionTreeClassifier(max_depth=25, min_samples_split=3, min_samples_leaf=1, random_state=42),
            False,
            'Interpretable decision tree classifier'
        ),
        'Random Forest Classifier': (
            RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1),
            False,
            'Tuned ensemble random forest'
        ),
        'XGBoost Classifier': (
            XGBClassifier(n_estimators=900, max_depth=16, learning_rate=0.025, subsample=0.88, colsample_bytree=0.88, gamma=0.05, random_state=42, eval_metric='logloss'),
            False,
            'Tuned gradient boosted decision trees'
        )
    }
    return models

build_model_dictionary = get_core_models
