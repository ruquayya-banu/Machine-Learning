Supervised Learning Workflows — From Beginner to Confident
# 1. Understand the Master Workflow

Supervised learning means teaching a machine learning model using examples that already have correct answers.

Example Dataset

Suppose we want to predict whether a student will pass an exam based on the number of hours studied.

Study Hours (X)	Result (y)   \
            1	Fail            \
            2	Fail\
            5	Pass\
            7	Pass\
            8	Pass

#### Important concepts:
- Features (X): Input information used to make predictions.
- Target (y): The answer the model learns to predict.
- Training: Teaching the model using examples and their correct answers.
- Prediction: Using the trained model to predict an answer for new data.
# 2. Memorize These 7 Workflow Steps
Follow these steps for most supervised learning algorithms.

a) Import — Import the required Python libraries.\
b) Load — Load the dataset.\
c) Separate — Separate features (X) and target (y).\
d) Split — Divide the data into training and testing sets.\
e) Create — Create the machine learning model.\
f) Fit and Predict — Train the model and make predictions.\
g) Evaluate — Evaluate the model's performance.

Memory Trick\
Import → Load → Separate → Split → Create → Fit/Predict → Evaluate\

Remember this common workflow first. Then learn what changes for each algorithm.

# 3. Build Your First Model: Logistic Regression

Logistic Regression is a supervised learning algorithm commonly used for classification.

In this example, we will predict whether a student passes or fails based on study hours.

Step 1: Import Libraries
python
```
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

What do these libraries do?

- pandas: Creates and manages tabular data.
- train_test_split: Splits data into training and testing sets.
- LogisticRegression: Creates the classification model.
- accuracy_score: Measures the proportion of correct predictions.
Step 2: Create the Dataset

We will create a small example dataset instead of loading a CSV file.

python
```
df = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
              4, 5, 6, 7, 8, 1, 3, 5, 6, 8],
    "result": ["Fail", "Fail", "Fail", "Fail", "Pass",
               "Pass", "Pass", "Pass", "Fail", "Fail",
               "Fail", "Pass", "Pass", "Pass", "Pass",
               "Fail", "Fail", "Pass", "Pass", "Pass"]
})

print(df.head())
```
Explanation:

hours contains the number of hours studied.
result contains the student's outcome.
df is the DataFrame containing the dataset.
Step 3: Separate Features and Target
python
```
X = df[["hours"]]
y = df["result"]
```

Explanation:

X contains the input features.
y contains the target we want to predict.

Memory trick: X is the question; y is the answer.

Note: Double brackets in df[["hours"]] keep X as a DataFrame.

# Step 4: Split the Dataset
python
```
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Explanation:

- X_train: Features used to train the model.
- X_test: Features used to test the model.
- y_train: Correct answers used during training.
- y_test: Correct answers used for evaluation.
- test_size=0.2: Reserves 20% of the data for testing.
- random_state=42: Makes the split reproducible.
- stratify=y: Helps preserve class proportions in both sets.
# Step 5: Create the Model
python
```
model = LogisticRegression()
```

This creates a Logistic Regression model.

At this point, the model has not learned from the dataset.

# Step 6: Train the Model
python
```
model.fit(X_train, y_train)
```

The fit() method trains the model using the training features and their correct answers.

Remember: fit() means learn.

# Step 7: Make Predictions
python
```
y_pred = model.predict(X_test)

print(y_pred)
```

The predict() method predicts the outcomes for the test examples.

Remember: predict() means produce predictions.

# Step 8: Evaluate the Model
python
```
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

Accuracy measures the proportion of predictions that are correct.

For example, if a model correctly predicts 4 out of 5 examples:

Accuracy = 4 / 5 = 0.80 = 80%

The actual accuracy depends on the test data and the model's predictions.

# 4. Complete Python Workflow

Here is the complete code in one place.

python
```
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Create the dataset
df = pd.DataFrame({
    "hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
        4, 5, 6, 7, 8, 1, 3, 5, 6, 8
    ],
    "result": [
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Fail", "Fail",
        "Fail", "Pass", "Pass", "Pass", "Pass",
        "Fail", "Fail", "Pass", "Pass", "Pass"
    ]
})

# 3. Separate features and target
X = df[["hours"]]
y = df["result"]

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create the model
model = LogisticRegression()

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 9. Predict for a new student
new_student = pd.DataFrame({"hours": [6]})

prediction = model.predict(new_student)
print("Predicted result:", prediction[0])
```

Note: The workflow has seven main steps, with dataset creation shown as the loading step for this example. Predicting for a new student is an optional additional step.

5. Important Things to Remember
Method or Variable	Purpose
X	Input features
y	Target variable
train_test_split()	Divides data into training and testing sets
model = LogisticRegression()	Creates the model
model.fit()	 Trains the model
model.predict()	Generates predictions
accuracy_score()	Evaluates classification accuracy
Final Memory Trick

Import → Load → Separate → Split → Create → Fit → Predict → Evaluate

This workflow is the foundation for learning other supervised learning algorithms, including Linear Regression, Decision Trees, Random Forest, KNN, SVM, and Naive Bayes.

The example dataset is only for learning the workflow. Real-world projects require appropriate data preparation, validation, and evaluation metrics.
