# aviacompany-clients

 _this project is dedicated to fully observe the aviacompany clients research and how they are satisfied in accordance with some parameters_

Firstly, we  imported the necessary libraries - now we can proceed our dataset using these them.
 
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

We mentioning our dataset path:

```
DATASET_PATH = "https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv"
```
## **Data load and data analysis**
###  loading...

We loaded the data from the 'clients.csv' file into a pandas DataFrame using the read_csv() function. 
```
df = pd.read_csv('clients.csv')
df.head()
```
## **Data description**
###  target variable 

- `satisfaction`: удовлетворенность клиента полетом, бинарная (*satisfied* или *neutral or dissatisfied*)

**Features (in Russian)**
- `Gender` (categorical: _Male_ или _Female_): пол клиента
- `Age` (numeric, int): количество полных лет
- `Customer Type` (categorical: _Loyal Customer_ или _disloyal Customer_): лоялен ли клиент авиакомпании?
- `Type of Travel` (categorical: _Business travel_ или _Personal Travel_): тип поездки
- `Class` (categorical: _Business_ или _Eco_, или _Eco Plus_): класс обслуживания в самолете
- `Flight Distance` (numeric, int): дальность перелета (в милях)
- `Departure Delay in Minutes` (numeric, int): задержка отправления (неотрицательная)
- `Arrival Delay in Minutes` (numeric, int): задержка прибытия (неотрицательная)
- `Inflight wifi service` (categorical, int): оценка клиентом интернета на борту
- `Departure/Arrival time convenient` (categorical, int): оценка клиентом удобство времени прилета и вылета
- `Ease of Online booking` (categorical, int): оценка клиентом удобства онлайн-бронирования
- `Gate location` (categorical, int): оценка клиентом расположения выхода на посадку в аэропорту
- `Food and drink` (categorical, int): оценка клиентом еды и напитков на борту
- `Online boarding` (categorical, int): оценка клиентом выбора места в самолете
- `Seat comfort` (categorical, int): оценка клиентом удобства сиденья
- `Inflight entertainment` (categorical, int): оценка клиентом развлечений на борту
- `On-board service` (categorical, int): оценка клиентом обслуживания на борту
- `Leg room service` (categorical, int): оценка клиентом места в ногах на борту
- `Baggage handling` (categorical, int): оценка клиентом обращения с багажом
- `Checkin service` (categorical, int): оценка клиентом регистрации на рейс
- `Inflight service` (categorical, int): оценка клиентом обслуживания на борту
- `Cleanliness` (categorical, int): оценка клиентом чистоты на борту


We provide a summary of the DataFrame's structure and information about its columns.
```
#dataset information
DATASET_PATH = # информация от столбцах
df.info()
```
Now we are getting the dimensions of our DataFrame. 
Also we are getting descriptionabout  information such as the count, mean, standard deviation, minimum value, quartiles, and maximum value for each numerical column.
```
df.shape
df.describe
```
Specifying our DataFrame:
```
df[['Flight Distance','Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Checkin service']].describe()
```
Getting sure where we need to fill the empty spaces.
```
print(df.isna().any())
print(df.isna().sum())
missing_values = df.isnull().sum()

# filling with fillna()

data_filled = df.fillna(0)
print("empty space in each column:")
print(missing_values)
print(data_filled)
```

## EDA 
Firtly, we create each subplot represents a different classes. 
Next - the Flight Distance plot.
```
#classes
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for i in range(3):
    axs[i].hist(df[df["id"]==i+1]["Age"].dropna(), density=True)
    axs[i].set_xlabel("Cleanliness")
    axs[i].set_ylabel("Seat comfort")
    axs[i].set_title("Class {}".format(i+1))
```
```
#flightdistance

plt.show()

sns.histplot(df['Flight Distance'], kde=True)
plt.show()
```

```
#customertype number statistics

df['Customer Type'].value_counts(normalize=True).plot(kind='bar', rot=60);
```
```
#seat comfort along the 5 metric

plot = sns.displot(df['Seat comfort'])
plot.set(xlim=(0, 5))
```
```
#inflight entertainment in plot!
sns.histplot(df['Inflight entertainment'], kde=True)
plt.show()
```

```
#correlation matrix for the columns

corr = df[['Flight Distance','Arrival Delay in Minutes', 'Departure Delay in Minutes']].corr()
sns.heatmap(corr, cmap="crest")
```

```
object_columns = df.select_dtypes(include=['object']).columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

df[object_columns] = df[object_columns].fillna('NaN')
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# checking up 
null_values = df.isnull().sum()
total_values = df.shape[0]
null_percentages = null_values / total_values * 100

# results
print("Количество и доля пропусков в данных после заполнения:")
print(null_values)
print(null_percentages)
```
Defining the target column and the features:
```
y = df['satisfaction']
X.head()
```
Splitting on train and test sets - checking out the dimension.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.shape, X_test.shape
```
Scaling features:
```
from sklearn.preprocessing import StandardScaler
st_X= StandardScaler()
X_train= st_X.fit_transform(X_train)
X_test= st_X.transform(X_test)
```
Working on categorical variables:
```
from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values

lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

#view transformed values

print(y_transformed)
```
```
from sklearn.impute import SimpleImputer

# Create an instance of SimpleImputer with mean strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data
imputer.fit(X_train)

# Transform the training and testing data
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

Now we are building the Linear Regression and estimatimg the error  of it:
```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create an instance of the LinearRegression model
model = LinearRegression()

# fit the model to the training data
model.fit(X_train, y_train)

# predict on the test data
y_pred = model.predict(X_test)

# evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Mean Squared Error: 1.953085
```
Now we are use StandardScaler and make the feature scaling:
```
from sklearn.preprocessing import StandardScaler
st_X= StandardScaler()
X_train= st_X.fit_transform(X_train)
X_test= st_X.transform(X_test)
```
```
X = df.drop(['Departure/Arrival time convenient', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction', 'Gender'], axis=1)
y = df['satisfaction']
X.head()
from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values

lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

#view transformed values

print(y_transformed)
```
Estimating the classifier of the regression:
```
#fit logistic regression model
classifier = LogisticRegression()
classifier.fit(X, y_transformed)
```
Estimating the accuracy of the regression:
```
from sklearn.metrics import accuracy_score

# predict the target variable for the input features
y_pred = classifier.predict(X)

# calculate the accuracy of the model
accuracy = accuracy_score(y_transformed, y_pred)

# print the accuracy
print("Accuracy:", accuracy)

#Accuracy: 0.2478
```
Now we are working on the categorical and on numerical variables:
```
num_cols = ['id', 'Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']

#categorical variables are changed
for col in cat_cols:
     df[col] = df[col].map(
        dict(zip(df[col].value_counts().index, range(len(df[col].value_counts().index))))
    )
df.head()
```
Catboost moment:
```
!pip install catboos

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

X = df.drop('satisfaction', axis=1)  
y = df['satisfaction']
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
Trying out the RandomForest then:
```
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=['satisfaction', 'id'])
y = df['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

 ## guess this work may be used in a experimnetal way so it could be researched more as far as i will stidy these new libraries! _

