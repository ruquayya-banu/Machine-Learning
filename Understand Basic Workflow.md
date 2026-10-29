Supervised Learning Workflows — From Beginner to Confident
Lesson 1: Understand the Master Workflow

Supervised learning means teaching a model using examples that already have correct answers.

For example, imagine a dataset that predicts whether a student will pass an exam based on study hours.

Study Hours (X)	Result (y)
1	Fail
2	Fail
5	Pass
7	Pass
8	Pass
Features (X): The information used to make predictions.
Target (y): The answer the model must learn to predict.
Training: Showing the model examples and their correct answers.
Prediction: Asking the model to predict an answer for new examples.
Lesson 2: Memorize These 7 Steps
Import: Bring in the required Python tools.
Load: Read your dataset.
Separate: Split data into X (features) and y (target).
Split: Create training and testing datasets.
Create: Choose a machine learning algorithm.
Fit and Predict: Train the model and generate predictions.
Evaluate: Check how accurate or useful the predictions are.
Memory Trick

Import → Load → Separate → Split → Create → Fit/Predict → Evaluate

Every time you study a new algorithm, start with these steps. Then learn only the differences.

Lesson 3: Write Your First Complete Python Workflow

We'll start with Logistic Regression, a classification algorithm. Our goal is to predict whether a student will pass or fail.

Step 1 — Import the Tools
python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


What each import does:

pandas: Handles tabular data.
train_test_split: Divides data into training and testing sets.
LogisticRegression: The algorithm we'll train.
accuracy_score: Measures the proportion of correct predictions.
Step 2 — Load the Data

For this first lesson, we'll create a small example dataset instead of loading a CSV file.

python
df = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
              4, 5, 6, 7, 8, 1, 3, 5, 6, 8],
    "result": ["Fail", "Fail", "Fail", "Fail", "Pass",
               "Pass", "Pass", "Pass", "Fail", "Fail",
               "Fail", "Pass", "Pass", "Pass", "Pass",
               "Fail", "Fail", "Pass", "Pass", "Pass"]
})

print(df.head())


df is our dataset. The hours column contains the study hours, and result contains the outcome.

Step 3 — Separate X and y
python
X = df[["hours"]]
y = df["result"]


This is one of the most important patterns to remember.

X = input features. Notice the double brackets because we want a DataFrame containing the hours column.
y = target or answer we want to predict.

Memory trick: X is the question; y is the answer.

Step 4 — Split the Data
python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


What does this mean?

X_train: Input examples used for training.
X_test: Input examples reserved for testing.
y_train: Correct answers used during training.
y_test: Correct answers used to evaluate the model.
test_size=0.2: Reserves 20% of the data for testing.
random_state=42: Makes the split reproducible.
stratify=y: Helps preserve the proportion of Pass and Fail examples.
Step 5 — Create the Model
python
model = LogisticRegression()


This line creates the algorithm. It has not learned from the data yet.

Think of model as a student who is ready to learn.

Step 6 — Train the Model
python
model.fit(X_train, y_train)


This is where learning happens.

The model studies the training examples and their known outcomes to learn a relationship between study hours and the result.

Remember: fit() means learn.

Step 7 — Make Predictions
python
y_pred = model.predict(X_test)

print(y_pred)


The model predicts Pass or Fail for each example in X_test.

Remember: predict() means produce answers.

Step 8 — Evaluate the Model
python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


Accuracy is the number of correct predictions divided by the total number of predictions.

For example, if the model correctly predicts 4 out of 5 students:

Accuracy
=
4
5
=
0.80
=
80
%

Your actual result depends on the test split and model predictions. This tiny, artificial dataset is for learning the workflow, not for building a reliable pass-prediction system.

Lesson 4: See the Entire Workflow Together

Here is the complete code to practice typing from memory.

python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
              4, 5, 6, 7, 8, 1, 3, 5, 6, 8],
    "result": ["Fail", "Fail", "Fail", "Fail", "Pass",
               "Pass", "Pass", "Pass", "Fail", "Fail",
               "Fail", "Pass", "Pass", "Pass", "Pass",
               "Fail", "Fail", "Pass", "Pass", "Pass"]
})

# 2. Separate features and target
X = df[["hours"]]
y = df["result"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create the model
model = LogisticRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Predict for a new example
new_student = pd.DataFrame({"hours": [6]})
print("Predicted result:", model.predict(new_student)[0])


Notice that we have seven main workflow steps, plus an optional step to predict for a new student.
