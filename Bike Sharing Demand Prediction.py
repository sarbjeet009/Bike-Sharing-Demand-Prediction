from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Split data into features and target variable
X = df.drop(columns=['cnt', 'casual', 'registered'])
y = df['cnt']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
print(f'R-squared on test set: {r2}')
