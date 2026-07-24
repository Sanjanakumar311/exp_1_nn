# exp_1_nn
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
