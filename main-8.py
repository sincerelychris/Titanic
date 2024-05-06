import pandas as pd

#loads data into repit
data = pd.read_csv('train.csv')

#display first few rows
print(data.head())

#display a summary and info about data
print(data.describe())
print(data.info())

#calculate fanily average
# Calculate the average family size

#average_family_size = (data['SibSp'] + data['Parch']).mean()
#print("Average family size on the Titanic was:", average_family_size)

#fill missing "age" values with the median values

# Fill missing 'Age' values with the median age
data['Age'] = data['Age'].fillna(data['Age'].median())

# Drop the 'Cabin' column as it has a lot of missing values
data.drop('Cabin', axis=1, inplace=True)

#convert sex to numerical values
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

#creating dummy variables

embarked_dummies = pd.get_dummies(data["Embarked"], prefix="Embarked")
data = pd.concat([data,embarked_dummies], axis=1)
data.drop("Embarked", axis=1, inplace=True)

# Drop features that are less likely to be directly useful
data.drop(["Ticket", "Name", "PassengerId"], axis=1, inplace=True)

print(data.head())