# Supervised Machine Learning Algorithms

Python implementations of common supervised learning algorithms using Scikit-learn.

## Classification Algorithms

### 1. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x_train, y_train)
```

### 2. K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(x_train, y_train)
```

### 3. Support Vector Machine (SVM)
```python
from sklearn.svm import SVC
SVM = SVC()
SVM.fit(x_train, y_train)
```

### 4. Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
```

### 5. Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
```

### 6. Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train, y_train)
```

### 7. AdaBoost Classifier
```python
from sklearn.ensemble import AdaBoostClassifier
AB = AdaBoostClassifier()
AB.fit(x_train, y_train)
```

### 8. Gradient Boosting Classifier
```python
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier()
GB.fit(x_train, y_train)
```

### 9. XGBoost Classifier
```python
from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(x_train, y_train)
```

### 10. Extra Trees Classifier
```python
from sklearn.ensemble import ExtraTreesClassifier
ET = ExtraTreesClassifier()
ET.fit(x_train, y_train)
```

### 11. Linear Discriminant Analysis (LDA)
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
LDA.fit(x_train, y_train)
```

### 12. Quadratic Discriminant Analysis (QDA)
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
QDA = QuadraticDiscriminantAnalysis()
QDA.fit(x_train, y_train)
```

### 13. Perceptron
```python
from sklearn.linear_model import Perceptron
P = Perceptron()
P.fit(x_train, y_train)
```

### 14. Stochastic Gradient Descent Classifier
```python
from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier()
SGD.fit(x_train, y_train)
```

### 15. Artificial Neural Network (MLP Classifier)
```python
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier()
MLP.fit(x_train, y_train)
```

## Regression Algorithms

### 16. Linear Regression
```python
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train, y_train)
```

### 17. Ridge Regression
```python
from sklearn.linear_model import Ridge
RR = Ridge()
RR.fit(x_train, y_train)
```

### 18. Lasso Regression
```python
from sklearn.linear_model import Lasso
LASSO = Lasso()
LASSO.fit(x_train, y_train)
```

### 19. ElasticNet Regression
```python
from sklearn.linear_model import ElasticNet
EN = ElasticNet()
EN.fit(x_train, y_train)
```

### 20. Decision Tree Regressor
```python
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()
DTR.fit(x_train, y_train)
```

### 21. Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()
RFR.fit(x_train, y_train)
```

### 22. Support Vector Regression (SVR)
```python
from sklearn.svm import SVR
SVR_model = SVR()
SVR_model.fit(x_train, y_train)
```

### 23. K-Neighbors Regressor
```python
from sklearn.neighbors import KNeighborsRegressor
KNR = KNeighborsRegressor()
KNR.fit(x_train, y_train)
```

### 24. Gradient Boosting Regressor
```python
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor()
GBR.fit(x_train, y_train)
```

### 25. AdaBoost Regressor
```python
from sklearn.ensemble import AdaBoostRegressor
ABR = AdaBoostRegressor()
ABR.fit(x_train, y_train)
```

### 26. Extra Trees Regressor
```python
from sklearn.ensemble import ExtraTreesRegressor
ETR = ExtraTreesRegressor()
ETR.fit(x_train, y_train)
```

### 27. XGBoost Regressor
```python
from xgboost import XGBRegressor
XGB = XGBRegressor()
XGB.fit(x_train, y_train)
```

### 28. Artificial Neural Network (MLP Regressor)
```python
from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor()
MLP.fit(x_train, y_train)
```

## Installation

```bash
pip install scikit-learn xgboost
```

## Notes

- `x_train` contains training features.
- `y_train` contains training target values.
- Classification predicts categorical classes.
- Regression predicts numerical values.
- Scale features when appropriate and tune hyperparameters to improve performance.
