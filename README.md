# Medical Insurance Cost Prediction

## Project Overview
This project aims to predict the medical insurance costs for individuals based on various features such as age, sex, BMI, number of children, smoking status, and region. Using a linear regression model, we analyze the dataset to understand how these features influence insurance charges and build a model to predict costs for new data points.

## Libraries and Dataset
We utilize several key Python libraries to load, analyze, and visualize the data:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```

Loading the dataset:
```python
df = pd.read_csv('insurance.csv')
df.head()
```

### Data Preview
| age | sex    | bmi  | children | smoker | region    | charges     |
|-----|--------|------|----------|--------|-----------|-------------|
| 19  | female | 27.9 | 0        | yes    | southwest | 16884.92400 |
| 18  | male   | 33.77| 1        | no     | southeast | 1725.55230  |
| 28  | male   | 33.0 | 3        | no     | southeast | 4449.46200  |
| 33  | male   | 22.705| 0       | no     | northwest | 21984.47061 |
| 32  | male   | 28.88| 0        | no     | northwest | 3866.85520  |

```python
df.shape
# Output: (1338, 7)
df.info()
# Detailed information about the dataset
df.describe()
# Statistical summary of the dataset
```

### Data Visualization
#### Age Distribution
```python
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['age'])
plt.title("Age Distribution")
plt.show()
```
![Age Distribution](https://github.com/mohammadquadri0004/python/assets/146173492/deb2ce92-3367-4d46-9ec1-fc4075c97e41)

#### Sex Distribution
```python
plt.figure(figsize=(6,6))
sns.countplot(x="sex",data=df)
plt.title("Sex Distribution")
plt.show()
```
![Sex Distribution](https://github.com/mohammadquadri0004/python/assets/146173492/0459818b-a212-467a-b230-529ba9685253)

#### BMI Distribution
```python
sns.distplot(df['bmi'])
plt.show()
```
![BMI Distribution](https://github.com/mohammadquadri0004/python/assets/146173492/9035e749-0eb1-49d9-8b34-48381dd49295)

### Data Preprocessing
We replace categorical variables with numerical values for easier model training:
```python
df.replace({'sex':{'male':0,'female':1}}, inplace=True)
df.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)
```

Splitting the data into features and target:
```python
X = df.drop(columns="charges", axis=1)
y = df['charges']
```

### Model Training
Splitting the dataset into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
Training the Linear Regression model:
```python
reg = LinearRegression()
reg.fit(X_train, y_train)
```
Evaluating the model:
```python
training_data_prediction = reg.predict(X_train)
r2_train = metrics.r2_score(y_train, training_data_prediction)
print("R-squared value for training data: ", r2_train)
# Output: 0.7537217150960405
```

### Prediction
Making predictions with the trained model:
```python
sample_input_data = (30, 1, 22.7, 0, 1, 0)
input_data_as_numpy_array = np.asarray(sample_input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = reg.predict(input_data_reshaped)
print("The insurance cost is ", prediction)
# Output: [2223.03197428]
```
