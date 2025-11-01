import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "mushroom_model.pkl")

def main():
  # Load data
  ucirepo = fetch_ucirepo(id=73)
  X = ucirepo.data.features
  y = ucirepo.data.targets.squeeze()

  # Handle missing data
  if 'stalk-root' in X.columns:
      X['stalk-root'] = X['stalk-root'].replace('?', 'unknown')

  # Preprocessing
  categorical_features = X.columns.tolist()

  preprocessor = ColumnTransformer([
      ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
  ])

  # Define model
  model = RandomForestClassifier(n_estimators=100, random_state=42)

  # Make a pipeline
  clf = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('model', model)
  ])

  # Train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Fit + predict
  clf.fit(X_train, y_train)
  preds = clf.predict(X_test)

  # Evaluate
  acc = accuracy_score(y_test, preds)
  print(f"Accuracy: {acc:.3f}")
  print(classification_report(y_test, preds))

  # Save model
  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
  joblib.dump(clf, MODEL_PATH)
  print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()