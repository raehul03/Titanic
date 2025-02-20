# Titanic
Step 1: Load and Explore the Data
First, load the required libraries and read the dataset.

python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('/mnt/data/train.csv')
test_df = pd.read_csv('/mnt/data/test.csv')
gender_submission = pd.read_csv('/mnt/data/gender_submission.csv')

# Display the first few rows of the train dataset
train_df.head()
Step 2: Data Understanding
Check for missing values, data types, and summary statistics.

python
Copy
Edit
# Check the shape of the datasets
print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)

# Check for missing values
print("\nMissing values in Train Data:\n", train_df.isnull().sum())
print("\nMissing values in Test Data:\n", test_df.isnull().sum())

# Summary statistics
train_df.describe()
Step 3: Data Visualization
Visualize distributions and relationships between features.

python
Copy
Edit
# Survival count
sns.countplot(data=train_df, x='Survived')
plt.title("Survival Count")
plt.show()

# Survival by gender
sns.countplot(data=train_df, x='Sex', hue='Survived')
plt.title("Survival Rate by Gender")
plt.show()

# Survival by passenger class
sns.countplot(data=train_df, x='Pclass', hue='Survived')
plt.title("Survival Rate by Passenger Class")
plt.show()
Step 4: Data Preprocessing
Handle missing values and encode categorical variables.

python
Copy
Edit
# Fill missing Age values with median
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# Fill missing Embarked values with most common value
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# Fill missing Fare values in test dataset
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Convert categorical variables to numeric
train_df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
test_df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

train_df.replace({'Embarked': {'C': 0, 'Q': 1, 'S': 2}}, inplace=True)
test_df.replace({'Embarked': {'C': 0, 'Q': 1, 'S': 2}}, inplace=True)

# Drop unnecessary columns
train_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
test_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Display processed train dataset
train_df.head()
Step 5: Train-Test Split
Split the dataset into training and validation sets.

python
Copy
Edit
from sklearn.model_selection import train_test_split

# Features and target variable
X = train_df.drop(columns=['Survived'])
y = train_df['Survived']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set Shape:", X_train.shape)
print("Validation Set Shape:", X_val.shape)
Step 6: Train a Machine Learning Model
Use a Random Forest Classifier for prediction.

python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
Step 7: Make Predictions on Test Data
Use the trained model to predict on the test dataset.

python
Copy
Edit
# Predict on the test dataset
test_predictions = model.predict(test_df)

# Create a submission file
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)

# Show first few rows of the submission file
submission.head()
Step 8: Save and Load the Model (Optional)
Save the trained model for future use.

python
Copy
Edit
import joblib

# Save the model
joblib.dump(model, 'titanic_model.pkl')

# Load the model (for future use)
loaded_model = joblib.load('titanic_model.pkl')

# Verify by making predictions again
loaded_model.predict(X_val[:5])
