from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


mnist = fetch_openml('mnist_784', as_frame=False)

X = mnist.data
y = mnist.target.astype('int')

X = X / 255.0  # Normalize pixel values to [0, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model started...")
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', verbose=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
joblib.dump(model, "mnist_model.pkl")
print("Model saved as mnist_model.pkl")