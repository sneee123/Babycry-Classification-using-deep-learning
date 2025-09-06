import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

duration = 5  
sr = 16000    

print("Recording will start in 2 seconds...")
sd.sleep(2000)
print("Recording now...")
audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
print("Recording complete.")


output_path = "test_recording.wav"
write(output_path, sr, audio)
print(f"Saved audio to {output_path}")


yamnet = hub.load('https://tfhub.dev/google/yamnet/1')


model = load_model("models/model4.h5")
class_names = np.load("features/y_encoder.npy", allow_pickle=True)


y, sr = librosa.load(output_path, sr=16000)
y = y[:sr * 10]  
y = y.astype(np.float32)
_, embeddings, _ = yamnet(y)

if embeddings.shape[0] == 0:
    print("Unable to extract meaningful audio embeddings.")
else:
    feature = np.mean(embeddings.numpy(), axis=0).reshape(1, -1) 
    probs = model.predict(feature)
    pred_class = np.argmax(probs)
    pred_label = class_names[pred_class]
    print(f"\nPredicted Cry Type: {pred_label}")
    print("Probabilities:")
    for i, cls in enumerate(class_names):
        print(f"  {cls:15s}: {probs[0][i]:.3f}")
