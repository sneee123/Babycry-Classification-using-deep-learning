import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight


X = np.load("features2/X.npy")
y = np.load("features2/y.npy")
class_names = np.load("features2/y_encoder.npy", allow_pickle=True)


num_classes = len(class_names)
y_cat = to_categorical(y, num_classes=num_classes)


X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat, random_state=42)


y_int = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_int), y=y_int)
class_weights = dict(enumerate(class_weights))




model = Sequential([
    Input(shape=(1024,)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


es = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=32,
          class_weight=class_weights,
          callbacks=[es])

os.makedirs("models", exist_ok=True)
model.save("models/model4.h5")
print("Model saved to models/model4.h5")
