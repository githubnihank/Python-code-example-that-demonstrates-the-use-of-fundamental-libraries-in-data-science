# Python-code-example-that-demonstrates-the-use-of-fundamental-libraries-in-data-science
NumPy: For numerical operations, especially with arrays. Pandas: For data manipulation and analysis with DataFrames. Matplotlib: For basic plotting and visualization. Seaborn: For more advanced and aesthetically pleasing statistical visualizations (built on Matplotlib). Scikit-learn)
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("--- Data Science Libraries Demo ---")
print("\n1. NumPy: Numerical Operations")
# Create a NumPy array
data_points = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
print(f"Original NumPy array: {data_points}")

# Perform some basic operations
mean_value = np.mean(data_points)
std_dev = np.std(data_points)
sum_of_squares = np.sum(data_points**2)

print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_dev}")
print(f"Sum of squares: {sum_of_squares}")

# Array slicing and reshaping
subset = data_points[2:6]
print(f"Subset (elements from index 2 to 5): {subset}")

reshaped_data = data_points.reshape(3, 3)
print(f"Reshaped to 3x3 matrix:\n{reshaped_data}")

print("\n2. Pandas: Data Manipulation and Analysis")
# Create a dictionary to hold our hypothetical student data
student_data = {
    'StudentID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Harry', 'Ivy', 'Jack'],
    'Math_Score': [85, 78, 92, 65, 88, 70, 95, 80, 72, 89],
    'Science_Score': [70, 85, 88, 60, 90, 75, 92, 82, 68, 91],
    'Hours_Studied': [5, 4, 6, 3, 5, 4, 7, 5, 3, 6],
    'Gender': ['F', 'M', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'M']
}

# Create a Pandas DataFrame
df = pd.DataFrame(student_data)
print("Original DataFrame:")
print(df)

# Display basic information about the DataFrame
print("\nDataFrame Info:")
df.info()

# Get descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Select specific columns
print("\nMath Scores:")
print(df['Math_Score'])

# Filter data
print("\nStudents with Math Score > 85:")
print(df[df['Math_Score'] > 85])

# Group by 'Gender' and calculate average scores
print("\nAverage Scores by Gender:")
print(df.groupby('Gender')[['Math_Score', 'Science_Score']].mean())

print("\n3. Matplotlib: Basic Plotting")
# Scatter plot: Math Score vs Hours Studied
plt.figure(figsize=(8, 5))
plt.scatter(df['Hours_Studied'], df['Math_Score'], color='blue', alpha=0.7)
plt.title('Math Score vs. Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Math Score')
plt.grid(True)
plt.show()

# Histogram of Math Scores
plt.figure(figsize=(8, 5))
plt.hist(df['Math_Score'], bins=5, color='green', edgecolor='black', alpha=0.7)
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

print("\n4. Seaborn: Enhanced Statistical Visualizations")
# Pairplot to see relationships between numerical features
# (Note: This might take a moment to generate depending on data size)
print("Generating Pairplot (showing relationships between numerical features)...")
sns.pairplot(df[['Math_Score', 'Science_Score', 'Hours_Studied']])
plt.suptitle('Pairplot of Student Performance Data', y=1.02) # Adjust title position
plt.show()

# Box plot of Math Score by Gender
plt.figure(figsize=(7, 5))
sns.boxplot(x='Gender', y='Math_Score', data=df)
plt.title('Math Score Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Math Score')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df[['Math_Score', 'Science_Score', 'Hours_Studied']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Scores and Hours Studied')
plt.show()

print("\n5. Scikit-learn: Basic Machine Learning")
# Objective: Predict Math_Score based on Hours_Studied and Science_Score
X = df[['Hours_Studied', 'Science_Score']] # Features
y = df['Math_Score'] # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape (X_train): {X_train.shape}")
print(f"Testing data shape (X_test): {X_test.shape}")

# Create a Linear Regression model
model = LinearRegression()

# Train the model
print("\nTraining Linear Regression model...")
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.interce
print("\n--- Demo Complete ---")


# OUTPUE
--- Data Science Libraries Demo ---

1. NumPy: Numerical Operations
Original NumPy array: [10 15 20 25 30 35 40 45 50]
Mean: 30.0
Standard Deviation: 12.909944487358056
Sum of squares: 9600
Subset (elements from index 2 to 5): [20 25 30 35]
Reshaped to 3x3 matrix:
[[10 15 20]
 [25 30 35]
 [40 45 50]]

2. Pandas: Data Manipulation and Analysis
Original DataFrame:
   StudentID     Name  Math_Score  Science_Score  Hours_Studied Gender
0          1    Alice          85             70              5      F
1          2      Bob          78             85              4      M
2          3  Charlie          92             88              6      M
3          4    David          65             60              3      M
4          5      Eve          88             90              5      F
5          6    Frank          70             75              4      M
6          7    Grace          95             92              7      F
7          8    Harry          80             82              5      M
8          9      Ivy          72             68              3      F
9         10     Jack          89             91              6      M

DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 6 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   StudentID      10 non-null     int64 
 1   Name           10 non-null     object
 2   Math_Score     10 non-null     int64 
 3   Science_Score  10 non-null     int64 
 4   Hours_Studied  10 non-null     int64 
 5   Gender         10 non-null     object
dtypes: int64(4), object(2)
memory usage: 612.0+ bytes

Descriptive Statistics:
       StudentID  Math_Score  Science_Score  Hours_Studied
count   10.00000   10.000000      10.000000      10.000000
mean     5.50000   81.400000      80.100000       4.800000
std      3.02765   10.068653      11.189777       1.316561
min      1.00000   65.000000      60.000000       3.000000
25%      3.25000   73.500000      71.250000       4.000000
50%      5.50000   82.500000      83.500000       5.000000
75%      7.75000   88.750000      89.500000       5.750000
max     10.00000   95.000000      92.000000       7.000000

Math Scores:
0    85
1    78
2    92
3    65
4    88
5    70
6    95
7    80
8    72
9    89
Name: Math_Score, dtype: int64

Students with Math Score > 85:
   StudentID     Name  Math_Score  Science_Score  Hours_Studied Gender
2          3  Charlie          92             88              6      M
4          5      Eve          88             90              5      F
6          7    Grace          95             92              7      F
9         10     Jack          89             91              6      M

Average Scores by Gender:
        Math_Score  Science_Score
Gender                           
F             85.0      80.000000
M             79.0      80.166667



![image](https://github.com/user-attachments/assets/75abd53c-14c6-44fd-96ff-cbe6851d3166)

![image](https://github.com/user-attachments/assets/bac7db78-5bcd-4ff2-9367-366511bcf5fc)
