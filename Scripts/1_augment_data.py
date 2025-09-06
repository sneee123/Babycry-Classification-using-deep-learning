import os
import librosa
import numpy as np
import soundfile as sf
import random
import csv
from datetime import datetime


AUGMENT_TARGET_COUNT = 1600
AUGMENT_COUNT_PER_FILE = 170
SEED = 42
random.seed(SEED)

def augment_randomly(y, sr):
    """Apply a random audio augmentation and return label for logging."""
    choice = random.choice(['time_stretch', 'pitch', 'noise', 'shift', 'reverb'])

    if choice == 'time_stretch':
        rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y=y, rate=rate), f"time_stretch({rate:.2f})"

    elif choice == 'pitch':
        steps = random.randint(-2, 2)
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps), f"pitch({steps})"

    elif choice == 'noise':
        noise = np.random.normal(0, 0.005, y.shape)
        return y + noise, "noise"

    elif choice == 'shift':
        shift = int(random.uniform(-0.1, 0.1) * sr)
        return np.roll(y, shift), f"shift({shift})"

    elif choice == 'reverb':
        reverb = np.convolve(y, np.random.randn(100) * 0.01, mode='full')[:len(y)]
        return reverb, "reverb"


def count_audio_files(directory):
    return len([f for f in os.listdir(directory) if f.endswith('.wav')])

def ensure_clipped(y):
    return np.clip(y, -1.0, 1.0)

def remove_silence_and_noise(y, sr, top_db=20):
  
    intervals = librosa.effects.split(y, top_db=top_db)
    nonsilent_audio = np.concatenate([y[start:end] for start, end in intervals])

    
    fft = np.fft.rfft(nonsilent_audio)
    freqs = np.fft.rfftfreq(len(nonsilent_audio), 1/sr)
    
    
    fft[freqs < 60] = 0
    denoised = np.fft.irfft(fft)

    return denoised

def smart_augment_dataset(input_dir, output_dir, metadata_path="augmentation_log.csv"):
    os.makedirs(output_dir, exist_ok=True)

    
    with open(metadata_path, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["timestamp", "class", "filename", "augmentation", "original_file"])

        for class_label in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_label)
            out_class_path = os.path.join(output_dir, class_label)

            if not os.path.isdir(class_path):
                print(f"Skipping {class_path} (not a folder)")
                continue

            os.makedirs(out_class_path, exist_ok=True)

            existing_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            n_existing = len(existing_files)

            print(f"\nClass '{class_label}' has {n_existing} original files.")

        
            for file in existing_files:
                src_path = os.path.join(class_path, file)
                dst_path = os.path.join(out_class_path, file)
                y, sr = librosa.load(src_path, sr=None)
                sf.write(dst_path, ensure_clipped(y), sr)

            if n_existing >= AUGMENT_TARGET_COUNT:
                continue

            needed_augments = AUGMENT_TARGET_COUNT - n_existing
            augments_per_file = max(1, min(AUGMENT_COUNT_PER_FILE, needed_augments // n_existing + 1))

            print(f"Augmenting class '{class_label}' with approx {augments_per_file}x per file")

            for file in existing_files:
                if needed_augments <= 0:
                    break

                file_path = os.path.join(class_path, file)
                y, sr = librosa.load(file_path, sr=None)
                y = remove_silence_and_noise(y, sr)


                for i in range(augments_per_file):
                    if needed_augments <= 0:
                        break

                    y_aug, aug_type = augment_randomly(y, sr)
                    y_aug = ensure_clipped(y_aug)

                    out_name = f"{os.path.splitext(file)[0]}_aug{i}.wav"
                    out_path = os.path.join(out_class_path, out_name)
                    sf.write(out_path, y_aug, sr)

                    log_writer.writerow([datetime.now().isoformat(), class_label, out_name, aug_type, file])
                    needed_augments -= 1

            print(f"Finished class '{class_label}' with total ~{count_audio_files(out_class_path)} files.")


smart_augment_dataset("audio_data", "balanced_data2", metadata_path="augmentation_log.csv")
