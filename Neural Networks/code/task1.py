import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
train_data = pd.read_csv('C:/Users/Samin/Desktop/University/Term 7/Computational Intelligence/Projects/Neural Networks/code/datasets/california_houses_price/california_housing_train.csv')
test_data = pd.read_csv('C:/Users/Samin/Desktop/University/Term 7/Computational Intelligence/Projects/Neural Networks/code/datasets/california_houses_price/california_housing_test.csv')

# Step 2: Explore the dataset
print(train_data.head())
print(test_data.head())

# Step 3: Split the features and labels
X_train = train_data.drop('median_house_value', axis=1)
y_train = train_data['median_house_value']

X_test = test_data.drop('median_house_value', axis=1)
y_test = test_data['median_house_value']

# Step 4: Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Verify the preprocessed data
print(X_train_scaled[:5])
print(X_test_scaled[:5])
