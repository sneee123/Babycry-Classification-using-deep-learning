import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import shutil
from datetime import datetime

# Parameters
duration = 5  # seconds
sr = 16000    # sampling rate

# Load YAMNet from TensorFlow Hub correctly as a KerasLayer
yamnet_layer = hub.KerasLayer('https://tfhub.dev/google/yamnet/1', trainable=False)
yamnet_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None,), dtype=tf.float32),
    yamnet_layer
])

# Load classifier model and class labels
model = load_model("models/model2.h5")
class_names = np.load("features/y_encoder.npy", allow_pickle=True)

# Step 1: Record audio
print("\nRecording in 2 seconds...")
sd.sleep(2000)
print("Recording...")
audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
write("test_recording.wav", sr, audio)
print("Saved: test_recording.wav")

# Step 2: Load and preprocess audio
y, _ = librosa.load("test_recording.wav", sr=sr)

if np.abs(y).mean() < 0.001:
    print("Silent or low-volume input detected. Skipping.")
    exit()

# YAMNet requires mono float32 audio sampled at 16kHz
y = y[:sr * 10]  # Trim to max 10s
y = y.astype(np.float32)

# Step 3: Extract YAMNet embeddings
scores, embeddings, spectrogram = yamnet_model(y)
feature = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)

# Step 4: Predict using your trained model
probs = model.predict(feature)
pred_idx = np.argmax(probs)
pred_label = class_names[pred_idx]

# Step 5: Show prediction
print(f"\nPrediction: {pred_label}")
for i, cls in enumerate(class_names):
    print(f"{cls:15s}: {probs[0][i]:.3f}")

# Step 6: Feedback loop
user_input = input("\nIs this correct? (y/n): ").lower()
if user_input == 'y':
    print("Great! Nothing to update.")
else:
    true_label = input("What is the correct label? (Options: " + ", ".join(class_names) + "): ").strip()
    if true_label not in class_names:
        print("Invalid label. Skipping update.")
    else:
        save_dir = os.path.join("corrected_data", true_label)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = os.path.join(save_dir, f"user_{timestamp}.wav")
        shutil.copy("test_recording.wav", dest_path)
        print(f"Saved corrected sample to {dest_path}")

        # Update correction count
        count_path = "corrected_data/.count"
        if not os.path.exists(count_path):
            count = 0
        else:
            with open(count_path, "r") as f:
                count = int(f.read().strip() or 0)

        count += 1
        with open(count_path, "w") as f:
            f.write(str(count))

        
        if count >= 5:
            print("Retraining after 5 corrections...")
            os.system("python Scripts/2_extract_features.py")
            os.system("python Scripts/3_train_model.py")
            with open(count_path, "w") as reset_f:
                reset_f.write("0")
            print("Done training!")
# Optional: retrain after 5 corrections