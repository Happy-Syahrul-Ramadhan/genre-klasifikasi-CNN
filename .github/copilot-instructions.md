# GitHub Copilot Instructions for Music Genre Classification Project

## Project Overview
This is a music genre classification system using Convolutional Neural Networks (CNNs) to classify audio into genres. The project compares two approaches:
1. **Audio-based classification**: Processing raw audio files (`.wav`) → MFCCs → CNN models
2. **Feature-based classification**: Using Spotify API tabular features → Logistic Regression/DNN

**Key Achievement**: Audio-based CNN models significantly outperform Spotify feature-based models (83.8% vs 53.2% accuracy).

## Architecture & Data Flow

### Three-Notebook Workflow
1. **`01_data_processing.ipynb`**: Audio collection → MFCC extraction → JSON serialization
2. **`02_modeling.ipynb`**: MFCC loading → Model training (DNN/CNN) → `.h5` model persistence
3. **`03_spotify_modeling.ipynb`**: Spotify API → Feature extraction → Baseline models

### Critical Data Paths (Windows absolute paths required)
```python
# All paths MUST use raw strings or escaped backslashes
path_data = r'D:\Music-Genre-Classification-Using-Convolutional-Neural-Networks-main\Music-Genre-Classification-Using-Convolutional-Neural-Networks-main\data\genres_original\'
json_path = r'D:\Music-Genre-Classification-Using-Convolutional-Neural-Networks-main\Music-Genre-Classification-Using-Convolutional-Neural-Networks-main\data\data.json'
```
**Never use relative paths like `../data/`** - they cause issues across notebooks.

## Audio Processing Pipeline

### MFCC Parameters (do not modify without retraining)
```python
fs = 22050              # Sampling rate (Hz)
n_mfcc = 13             # Number of MFCCs
n_fft = 2048            # FFT window size
hop_length = 512        # Hop size between frames
num_segments = 10       # Split 30s tracks → 3s segments
```

### Data Shape Requirements
- **DNN input**: `(n_samples, 130, 13)` → Flatten required
- **CNN input**: `(n_samples, 130, 13, 1)` → Add channel dimension
- **Output**: 10 genre classes (0-9 numeric encoding)

### Segment Creation Logic
Audio files are split into 3-second segments to increase training samples:
- 100 tracks/genre × 10 segments = 1000 samples/genre × 10 genres = 10,000 total samples
- Each segment: `(130 time steps, 13 MFCCs)`

## Model Architecture Patterns

### Progressive CNN Improvements (see `02_modeling.ipynb`)
1. **CNN-1** (69.1%): Base architecture, no regularization
2. **CNN-2** (80.1%): + Dropout + EarlyStopping
3. **CNN-3** (83.8%): + Data augmentation (horizontal flip = reversed audio)

### Key Regularization Strategy
- **Use Dropout** (0.3-0.5) at Dense layers only
- **Do NOT use L2 regularization on CNNs** - tested and degrades performance
- **EarlyStopping**: `patience=50, restore_best_weights=True`
- **Data Augmentation**: `horizontal_flip=True` (simulates reversed playback)

### Standard CNN Structure
```python
# Conv blocks: Conv2D → BatchNormalization → MaxPooling2D → Dropout
# Dense blocks: Flatten → Dense(Dropout) → Dense(output, softmax)
# Optimizer: Adam(learning_rate=0.001)
# Loss: sparse_categorical_crossentropy
```

## Development Workflows

### Running Notebooks Sequentially
```powershell
# 1. Process audio (slow: ~30 min for full dataset)
# Run all cells in 01_data_processing.ipynb
# Output: data/data.json (10,000 MFCC samples)

# 2. Train models (slow: ~1 hour per CNN with 250 epochs)
# Run all cells in 02_modeling.ipynb
# Output: models/model_*.h5 (4 models)

# 3. Spotify baseline (requires API credentials)
# Run all cells in 03_spotify_modeling.ipynb
```

### Model Persistence
Models saved as `.h5` files in `models/`:
- `model_dnn.h5` - Dense Neural Network baseline
- `model_cnn1.h5` - Base CNN without regularization
- `model_cnn2.h5` - CNN with dropout
- `model_cnn3.h5` - **Best model** with dropout + augmentation

Load with: `model = tf.keras.models.load_model('path/to/model.h5')`

## Dependencies & Environment

### Core Libraries (TensorFlow ecosystem)
```python
import librosa              # Audio processing (MFCC extraction)
import tensorflow.keras     # Model building
import numpy, pandas        # Data manipulation
import matplotlib.pyplot    # Visualization (spectrograms)
import scipy               # FFT calculations
```

### Spotify API Requirements (`03_spotify_modeling.ipynb` only)
- Install: `pip install spotipy`
- Requires: Client ID + Secret from Spotify Developer Dashboard
- API calls are rate-limited and slow (several days for 10,000 songs)

### Expected Data Structure
```
data/
  genres_original/     # Raw .wav files organized by genre
    blues/
    classical/
    ...
  data.json           # Generated: {"mfcc": [...], "genre_name": [...], "genre_num": [...]}
```

## Project-Specific Conventions

### Visualization Functions (defined in notebooks)
- `plot_waveform()` - Time-domain audio amplitude
- `plot_spec()` - Frequency spectrum (log scale)
- `plot_spectrogram()` - STFT time-frequency representation
- `plot_mel_spectrogram_audio()` - Mel-scaled spectrogram
- `plot_mfcc()` - MFCC visualization (final input format)

### Genre Mapping (numeric encoding)
```python
# Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
# Encoded: 0-9 (alphabetical order)
genre_map = dict(zip(sorted(set(genres)), np.arange(0, 10)))
```

### Train/Validation/Test Split
Standard split in `02_modeling.ipynb`:
- Training: 70%
- Validation: 15% (used during training)
- Test: 15% (held out for final evaluation)

## Common Issues & Solutions

### Path Issues on Windows
❌ `path = '../data/file.json'` (fails across notebook contexts)  
✅ `path = r'D:\...\data\file.json'` (absolute with raw string)

### Model Overfitting
All models overfit by design (validation loss > training loss). This is expected with CNNs on limited data. The goal is to balance:
- **Too little training**: Poor test accuracy
- **Too much training**: Severe overfitting (huge val/train gap)

Use early stopping to find optimal epoch count (typically 150-250 epochs).

### Memory Errors During Training
- Reduce batch size (default: 32)
- Process fewer segments per track
- Clear Keras session: `tf.keras.backend.clear_session()`

## Testing & Validation

### Model Evaluation Pattern
```python
# Predict on test set
y_pred = np.argmax(model.predict(X_test), axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

# Accuracy
accuracy = model.evaluate(X_test, y_test)[1]
```

### Performance Benchmarks
- **Random baseline**: 10% (10 classes)
- **Spotify Logistic Regression**: 53.2%
- **DNN on MFCCs**: 57.8%
- **Target CNN performance**: >80%

## Project Context
- **Research project**: Demonstrates audio ML capabilities
- **Production considerations**: Not production-ready (no CI/CD, no model versioning, hardcoded paths)
- **Dataset**: GTZAN genre classification (Kaggle)
- **Primary use case**: Genre classification for music recommendation systems
