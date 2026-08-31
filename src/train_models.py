"""
Model Training Definition Module
================================
STEP 5: Initializing the 7 core classifier algorithms.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

def get_core_models():
    """
    Returns dictionary of the exact 7 models and scaling settings.
    """
    models = {
        'XGBoost Classifier': (
            XGBClassifier(
                n_estimators=900, max_depth=16, learning_rate=0.025,
                subsample=0.88, colsample_bytree=0.88, gamma=0.05,
                random_state=42, eval_metric='logloss'
            ),
            False,
            'Peak high-accuracy gradient boosted classifier (90.1%)'
        ),
        'Random Forest Classifier': (
            RandomForestClassifier(
                n_estimators=500, max_depth=40, min_samples_split=2,
                min_samples_leaf=1, random_state=42, n_jobs=-1
            ),
            False,
            'Tuned ensemble random forest baseline (79.0%)'
        ),
        'Decision Tree Classifier': (
            DecisionTreeClassifier(
                max_depth=25, min_samples_split=3, min_samples_leaf=1,
                random_state=42
            ),
            False,
            'Interpretable decision tree classifier (78.1%)'
        ),
        'Linear Discriminant Analysis (LDA)': (
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            True,
            'Linear discriminant decision boundary (72.2%)'
        ),
        'Support Vector Machine (LinearSVC)': (
            LinearSVC(C=5.0, loss='squared_hinge', random_state=42, max_iter=4000),
            True,
            'Linear support vector classifier margin (72.1%)'
        ),
        'Logistic Regression': (
            LogisticRegression(C=10.0, solver='lbfgs', max_iter=4000, random_state=42),
            True,
            'Standard linear logistic regression (71.7%)'
        ),
        'Gaussian Naive Bayes': (
            GaussianNB(var_smoothing=1e-2),
            True,
            'Probabilistic naive bayes baseline (69.5%)'
        )
    }
    return models
