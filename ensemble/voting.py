from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier
# VotingClassifier
from sklearn.ensemble import VotingClassifier

# ensemble 할 model 정의
models = [
    ('ada', AdaBoostClassifier()),
    ('rfc', RandomForestClassifier()),
    ('knn', KNeighborsClassifier()),
    ('dtc', DecisionTreeClassifier()),
]

# hard vote
hard_vote  = VotingClassifier(models, voting='hard')
hard_vote_cv = cross_validate(hard_vote, x_train, y_train, cv=k_fold)
hard_vote.fit(x_train, y_train)

# soft vote
soft_vote  = VotingClassifier(models, voting='soft')
soft_vote_cv = cross_validate(soft_vote, x_train, y_train, cv=k_fold)
soft_vote.fit(x_train, y_train)