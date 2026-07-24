# exp_1_nn
<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
TYPE YOUR CODE HERE
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
data = pd.read_csv("Churn_Modelling (1).csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Dataset information
print("\nDataset Info:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Drop unnecessary columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical columns
le_geo = LabelEncoder()
le_gender = LabelEncoder()

data['Geography'] = le_geo.fit_transform(data['Geography'])
data['Gender'] = le_gender.fit_transform(data['Gender'])

print("\nDataset after Encoding:")
print(data.head())

# Separate features and target
X = data.drop('Exited', axis=1)
y = data['Exited']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\nNormalized Features (First 5 Rows):")
print(X_scaled[:5])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Display shapes
print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape :", X_test.shape)

## OUTPUT:
SHOW YOUR OUTPUT HERE
<img width="787" height="308" alt="image" src="https://github.com/user-attachments/assets/c15eff45-8e4b-407f-8c62-3f4302ca60f5" />
<img width="563" height="230" alt="image" src="https://github.com/user-attachments/assets/1948f430-cc79-4809-b815-e633be556a89" />
<img width="492" height="398" alt="image" src="https://github.com/user-attachments/assets/7d9353de-0fca-48c9-b188-d38917bd85f5" />
<img width="580" height="356" alt="image" src="https://github.com/user-attachments/assets/fb88c716-e42f-4412-bbb1-709a3b408b8b" />
<img width="797" height="417" alt="image" src="https://github.com/user-attachments/assets/5b01ac24-d6df-4f0c-970f-2ea4feca3e6b" />
<img width="837" height="368" alt="image" src="https://github.com/user-attachments/assets/31a68afb-2b4a-4159-9f2f-1701bdc51e6c" />
<img width="822" height="398" alt="image" src="https://github.com/user-attachments/assets/81ff5fee-0f4d-4066-91f2-14c0abdca1ee" />
<img width="721" height="297" alt="image" src="https://github.com/user-attachments/assets/5602dd09-b69d-4374-87b3-3c9bac76c1b5" />






## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


