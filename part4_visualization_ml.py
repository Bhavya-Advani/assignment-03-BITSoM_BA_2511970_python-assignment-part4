
pip install pd
import pandas as pd

df = pd.read_csv("/tmp/students.csv")

print("\n Data Exploration with Pandas")

print("First 5 rows:")
print(df.head())

print(f"\nShape: {df.shape}")
print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nPass/Fail counts:")
print(df['passed'].value_counts())

subject_cols = ['math', 'science', 'english', 'history', 'pe']
print("\nAverage scores per subject for passing students:")
print(df[df['passed'] == 1][subject_cols].mean())

print("\nAverage scores per subject for failing students:")
print(df[df['passed'] == 0][subject_cols].mean())

df['overall_avg'] = df[subject_cols].mean(axis=1)
top_student = df.loc[df['overall_avg'].idxmax()]
print(f"\nStudent with highest overall average: {top_student['name']} ({top_student['overall_avg']:.2f})")

## Data Visualization with Matplotlib

import matplotlib.pyplot as plt

df['avg_score'] = df[subject_cols].mean(axis=1)

# 1. Bar Chart — Average score per subject
plt.figure(figsize=(8, 5))
avg_scores = df[subject_cols].mean()
plt.bar(subject_cols, avg_scores, color='skyblue')
plt.title('Average Score per Subject Across All Students')
plt.xlabel('Subject')
plt.ylabel('Average Score')
plt.savefig('/tmp/plot1_bar.png')
plt.show()

# 2. Histogram — Distribution of math scores
plt.figure(figsize=(8, 5))
plt.hist(df['math'], bins=5, color='lightgreen', edgecolor='black')
mean_math = df['math'].mean()
plt.axvline(mean_math, color='red', linestyle='--', label=f'Mean: {mean_math:.2f}')
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('/tmp/plot2_histogram.png')
plt.show()

# 3. Scatter Plot — study_hours_per_day vs avg_score, color by passed
plt.figure(figsize=(8, 5))
pass_df = df[df['passed'] == 1]
fail_df = df[df['passed'] == 0]
plt.scatter(pass_df['study_hours_per_day'], pass_df['avg_score'], color='blue', label='Pass', alpha=0.7)
plt.scatter(fail_df['study_hours_per_day'], fail_df['avg_score'], color='red', label='Fail', alpha=0.7)
plt.title('Study Hours per Day vs Average Score')
plt.xlabel('Study Hours per Day')
plt.ylabel('Average Score')
plt.legend()
plt.savefig('/tmp/plot3_scatter.png')
plt.show()

# 4. Box Plot — attendance_pct for pass vs fail
plt.figure(figsize=(8, 5))
pass_attendance = df[df['passed'] == 1]['attendance_pct'].tolist()
fail_attendance = df[df['passed'] == 0]['attendance_pct'].tolist()
plt.boxplot([pass_attendance, fail_attendance], labels=['Pass', 'Fail'])
plt.title('Attendance Percentage Distribution')
plt.ylabel('Attendance Percentage')
plt.savefig('/tmp/plot4_box.png')
plt.show()

# 5. Line Plot — math and science scores for every student
plt.figure(figsize=(10, 6))
plt.plot(df['name'], df['math'], marker='o', label='Math', linestyle='-')
plt.plot(df['name'], df['science'], marker='s', label='Science', linestyle='--')
plt.title('Math and Science Scores for Each Student')
plt.xlabel('Student Name')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('/tmp/plot5_line.png')
plt.show()



## Data Visualization with Seaborn

import seaborn as sns

# 1. Bar plot: average math and science scores by passed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=df, x='passed', y='math', ax=ax1, palette='Blues')
ax1.set_title('Average Math Score by Pass/Fail')
ax1.set_xlabel('Passed')
ax1.set_ylabel('Math Score')

sns.barplot(data=df, x='passed', y='science', ax=ax2, palette='Greens')
ax2.set_title('Average Science Score by Pass/Fail')
ax2.set_xlabel('Passed')
ax2.set_ylabel('Science Score')

plt.tight_layout()
plt.savefig('/tmp/seaborn_bar.png')
plt.show()

# 2. Scatter plot with regression: attendance_pct vs avg_score, colored by passed
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='attendance_pct', y='avg_score', hue='passed', palette=['red', 'blue'])
sns.regplot(data=df[df['passed'] == 1], x='attendance_pct', y='avg_score', scatter=False, color='blue', label='Pass')
sns.regplot(data=df[df['passed'] == 0], x='attendance_pct', y='avg_score', scatter=False, color='red', label='Fail')
plt.title('Attendance Percentage vs Average Score with Regression Lines')
plt.xlabel('Attendance Percentage')
plt.ylabel('Average Score')
plt.legend()
plt.savefig('/tmp/seaborn_scatter.png')
plt.show()


## Machine Learning with scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

features = ['math', 'science', 'english', 'history', 'pe', 'attendance_pct', 'study_hours_per_day']
X = df[features]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
print(f"\nTraining Accuracy: {train_accuracy:.2f}")

y_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

print("\nTest Set Predictions:")
for i in range(len(X_test)):
    name = df.loc[X_test.index[i], 'name']
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    status = "✅ correct" if actual == predicted else "❌ wrong"
    print(f"{name}: Actual={actual}, Predicted={predicted} ({status})")

# Feature Importance
coefficients = model.coef_[0]
feature_importance = list(zip(features, coefficients))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nFeature Importance (sorted by absolute coefficient):")
for feature, coef in feature_importance:
    print(f"{feature}: {coef:.4f}")

# Horizontal bar chart
plt.figure(figsize=(10, 6))
colors = ['green' if coef > 0 else 'red' for _, coef in feature_importance]
plt.barh([f[0] for f in feature_importance], [f[1] for f in feature_importance], color=colors)
plt.title('Feature Coefficients in Logistic Regression')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.axvline(0, color='black', linewidth=0.8)
plt.savefig('/tmp/feature_importance.png')
plt.show()

# Predict for new student
new_student = [[75, 70, 68, 65, 80, 82, 3.2]]
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)[0]
probabilities = model.predict_proba(new_student_scaled)[0]
print(f"\nNew Student Prediction: {'Pass' if prediction == 1 else 'Fail'}")
print(f"Probabilities: Pass={probabilities[1]:.4f}, Fail={probabilities[0]:.4f}")



