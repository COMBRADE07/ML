import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# load data
iris = load_iris()
data = iris.feature_names
df = pd.DataFrame(iris.data, columns=data)
df['target'] = iris.target

x = df.iloc[:, 0:4]
y = df.iloc[:, -1]
# split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# setting up gridsearchcv for single algorithm
'''gridsearchcv = GridSearchCV(SVC(gamma='auto'),
                            {
                                'C':[1,10,15,20],
                                'kernel':['rbf','linear','poly']
                            },cv=5,return_train_score=False)
gridsearchcv.fit(x,y)

df = pd.DataFrame(gridsearchcv.cv_results_)

df = df[['param_C', 'param_kernel','mean_test_score']]
print(df)

# finding best score and parameter
score = gridsearchcv.best_score_
para = gridsearchcv.best_params_
print(f'best score is: {score}\nparameter: {para}')'''

# checking for multiple algorithms
model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear', 'poly']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10]
        }
    },
    'logestic_regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [1, 10, 20],
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 1, 2, 3]
        }
    }
}

score = []
for mn,mp in model_params.items():
    gsv = GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    gsv.fit(x,y)
    score.append({
        'model':mn,
        'best_score':gsv.best_score_,
        'best_params':gsv.best_params_
    })

sc = pd.DataFrame(data=score,columns=['model','best_score','best_params'])
print(sc)