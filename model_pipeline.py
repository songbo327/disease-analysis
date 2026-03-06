import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
path = './cardio_train.csv'
df = pd.read_csv(path, sep=';')

df.drop('id', axis=1, inplace=True)
df['age'] = df['age'] / 365.25

# Drop duplicates
df.drop_duplicates(inplace=True)

# Filter BP, unreasonable values
df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 150)]

# Split features
X = df.drop('cardio', axis=1)
y = df['cardio']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train baseline
print("Training Baseline...")
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)

# Evaluate baseline
y_pred_base = baseline.predict(X_test)
print("Baseline Report:")
print(classification_report(y_test, y_pred_base))

# Train model
print("Training First Model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred_model = model.predict(X_test)
print("Model Report:")
print(classification_report(y_test, y_pred_model))
