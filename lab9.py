import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('/home/user/Загрузки/titanic_prepared.csv') 

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
df = df.drop('Unnamed: 0', axis=1)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f'Decision Tree accuracy: {dt_accuracy}')
print(f'XGBoost accuracy: {xgb_accuracy}')
print(f'Logistic Regression accuracy: {lr_accuracy}')

feature_importances = dt_model.feature_importances_

sorted_indices = feature_importances.argsort()[::-1]

top_features = X_train.columns[sorted_indices[:2]]
print(f'Самые важные признаки: {top_features}')

dt_model_top_features = DecisionTreeClassifier()
dt_model_top_features.fit(X_train[top_features], y_train)
dt_predictions_top_features = dt_model_top_features.predict(X_test[top_features])
dt_accuracy_top_features = accuracy_score(y_test, dt_predictions_top_features)

print(f'Decision Tree (только самые важные признаки) accuracy: {dt_accuracy_top_features}')
