#audio to sepctrogram

import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
audio_file = 'path_to_audio_file.wav'
y, sr = librosa.load(audio_file, sr=None)

# Convert to Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert to log scale (log-magnitude)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

# Plotting the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency Spectrogram')
plt.show()


#data for CNN

import os
import librosa
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_audio(file_path, target_size=(64, 64)):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)  # Convert to log scale
    
    # Resize the spectrogram to fit the input size of the CNN (e.g., 64x64)
    log_mel_spectrogram_resized = librosa.util.fix_length(log_mel_spectrogram, size=target_size[0])
    
    # Convert to an image format (height x width x channels)
    return img_to_array(log_mel_spectrogram_resized)

# Example: Load a set of audio files from a directory
audio_dir = 'path_to_audio_dataset'
X = []
y = []

for class_dir in os.listdir(audio_dir):
    class_path = os.path.join(audio_dir, class_dir)
    for audio_file in os.listdir(class_path):
        file_path = os.path.join(class_path, audio_file)
        X.append(preprocess_audio(file_path))
        y.append(class_dir)  # Using class name as the label

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the data
X = X / 255.0  # Normalizing the spectrograms

# Encode labels if necessary
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)


#build  CNN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),  # Input shape is spectrogram size
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)


#evaluate
test_audio = 'path_to_test_audio.wav'
X_test = preprocess_audio(test_audio)
X_test = X_test.reshape(1, 64, 64, 1)  # Reshaping for CNN input
X_test = X_test / 255.0  # Normalize

# Predict
pred = model.predict(X_test)
pred_label = encoder.inverse_transform([np.argmax(pred)])
print(f"Predicted Label: {pred_label}")
