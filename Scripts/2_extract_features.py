import os
import numpy as np
import librosa
import tensorflow_hub as hub
from tqdm import tqdm

# Load YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_yamnet_embedding(wav_path):
    try:
        waveform, sr = librosa.load(wav_path, sr=16000)
        if len(waveform) < 16000: 
            print(f"Skipped (too short): {wav_path}")
            return None
        waveform = waveform[:16000 * 10] 
        waveform = waveform.astype(np.float32) 
       
        _, embeddings, _ = yamnet_model(waveform)
        if embeddings.shape[0] == 0:
            print(f"Empty embedding: {wav_path}")
            return None
        return np.mean(embeddings.numpy(), axis=0)  
    except Exception as e:
        print(f"Error: {e} in {wav_path}")
        return None



data_dir = "balanced_data2"
additional_dir = "corrected_data"

if os.path.exists(additional_dir):
    data_dir = "combined_temp"
    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.copytree("balanced_data2", data_dir)
    for label in os.listdir(additional_dir):
        label_path = os.path.join(additional_dir, label)
        if not os.path.isdir(label_path):
            continue  # skip files like .count

        os.makedirs(os.path.join(data_dir, label), exist_ok=True)
        for f in os.listdir(label_path):
            shutil.copy(os.path.join(label_path, f), os.path.join(data_dir, label, f))

X, y = [], []

for label in sorted(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, label)
    if not os.path.isdir(class_path):
        continue

    for file in tqdm(os.listdir(class_path), desc=f"Processing {label}"):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(class_path, file)
        emb = extract_yamnet_embedding(file_path)
        if emb is not None and len(emb) == 1024:
            X.append(emb)
            y.append(label)
        else:
            print(f"Skipped {file} (invalid embedding)")



from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
y_encoded = le.fit_transform(y)


os.makedirs("features2", exist_ok=True)
np.save("features2/X.npy", np.array(X))
np.save("features2/y.npy", np.array(y_encoded))
np.save("features2/y_encoder.npy", le.classes_)  

print("Saved:")
print("  features/X.npy")
print("  features/y.npy (encoded labels)")
print("  features/y_encoder.npy (label classes)")
