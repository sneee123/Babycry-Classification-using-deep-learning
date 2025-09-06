import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


X = np.load("features/X.npy")
y = np.load("features/y.npy")
class_names = np.load("features/y_encoder.npy", allow_pickle=True)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = load_model("models/model4.h5")


y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)


print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))


cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


 git --ver