import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tempfile
import os
import math

# Page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Title and description
st.title("🎵 Music Genre Classification")
st.markdown("""
This application uses a Convolutional Neural Network (CNN) to classify music into three genres:
**Ambient**, **Pop**, and **Rock**.

Upload an MP3 file and the model will predict its genre!
""")

# Model path - works for both local and production
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_cnn2.h5')

# Fallback paths for different deployment scenarios
if not os.path.exists(MODEL_PATH):
    # Try relative to streamlit folder
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model_cnn2.h5')
    if not os.path.exists(MODEL_PATH):
        # Try current directory
        MODEL_PATH = 'models/model_cnn2.h5'

# Genre labels
GENRES = ['ambient', 'pop', 'rock']

# Audio processing parameters
FS = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 30
NUM_SEGMENTS = 10


@st.cache_resource
def load_trained_model():
    """Load the pre-trained CNN model"""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def extract_mfcc_from_audio(file_path):
    """Extract MFCC features from audio file - matches preprocessing exactly"""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=FS, duration=DURATION)
        
        # Calculate samples per segment (SAME AS PREPROCESSING)
        samples_per_track = FS * DURATION
        samps_per_segment = int(samples_per_track / NUM_SEGMENTS)
        
        # Calculate expected MFCC length per segment (CRITICAL!)
        mfccs_per_segment = math.ceil(samps_per_segment / HOP_LENGTH)
        
        mfccs_list = []
        
        # Extract MFCC for each segment
        for seg in range(NUM_SEGMENTS):
            start_sample = seg * samps_per_segment
            end_sample = start_sample + samps_per_segment
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio[start_sample:end_sample],
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mfcc=N_MFCC
            )
            
            mfcc = mfcc.T  # Transpose to (time_steps, n_mfcc)
            
            # CRITICAL: Only append if correct length (SAME AS PREPROCESSING)
            if len(mfcc) == mfccs_per_segment:
                mfccs_list.append(mfcc)
        
        # Check if we got all segments
        if len(mfccs_list) != NUM_SEGMENTS:
            st.warning(f"⚠️ Only {len(mfccs_list)}/{NUM_SEGMENTS} segments have correct length. This may affect prediction accuracy.")
        
        return np.array(mfccs_list), audio, sr
    
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None, None


def predict_genre(model, mfccs):
    """Predict genre from MFCC features"""
    # Add channel dimension for CNN
    mfccs_cnn = mfccs[..., np.newaxis]
    
    # Predict for all segments
    predictions = model.predict(mfccs_cnn, verbose=0)
    
    # Average predictions across segments
    avg_prediction = np.mean(predictions, axis=0)
    
    # Get predicted class
    predicted_class = np.argmax(avg_prediction)
    confidence = avg_prediction[predicted_class] * 100
    
    return GENRES[predicted_class], confidence, avg_prediction


def plot_waveform(audio, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr, alpha=0.8, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    plt.tight_layout()
    return fig


def plot_spectrogram(audio, sr):
    """Plot spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Calculate STFT
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    
    img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='log', 
                                    cmap='viridis', ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(audio, sr):
    """Plot mel-scaled spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Calculate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, 
                                               n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel',
                                    cmap='viridis', ax=ax)
    ax.set_title('Mel-Scaled Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def plot_mfcc(mfcc, sr):
    """Plot MFCC features"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    img = librosa.display.specshow(mfcc.T, sr=sr, x_axis='time', 
                                    y_axis='linear', cmap='viridis', ax=ax)
    ax.set_title('MFCCs (Mel-Frequency Cepstral Coefficients)')
    ax.set_ylabel('MFCC Coefficient')
    fig.colorbar(img, ax=ax, format='%+2.0f')
    plt.tight_layout()
    return fig


def plot_prediction_bars(predictions):
    """Plot prediction probabilities as bar chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(GENRES, predictions * 100, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Confidence (%)', fontsize=12)
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_title('Genre Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# Main application
def main():
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header("📤 Upload Audio")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an MP3 file",
        type=['mp3'],
        help="Upload an MP3 file to classify its genre"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    **Model:** CNN with Regularization & Data Augmentation
    
    **Training Data:**
    - 3 Genres: Ambient, Pop, Rock
    - ~910 audio segments (3-second clips)
    
    **Features:**
    - 13 MFCCs (Mel-Frequency Cepstral Coefficients)
    - 10 segments per track
    """)
    
    # Main content
    if uploaded_file is not None:
        st.header("📊 Analysis Results")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Display file info
            st.subheader("📁 File Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Filename:** {uploaded_file.name}")
            with col2:
                st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")
            
            # Extract features
            with st.spinner("🎵 Processing audio and extracting features..."):
                mfccs, audio, sr = extract_mfcc_from_audio(tmp_file_path)
            
            if mfccs is not None:
                st.success("✅ Features extracted successfully!")
                
                # Make prediction
                with st.spinner("🤖 Predicting genre..."):
                    predicted_genre, confidence, all_predictions = predict_genre(model, mfccs)
                
                # Display prediction
                st.subheader("🎯 Genre Prediction")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Genre",
                        value=predicted_genre.upper(),
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1f}%",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        label="Segments Analyzed",
                        value=f"{len(mfccs)}",
                        delta=None
                    )
                
                # Prediction probabilities
                st.subheader("📈 Prediction Probabilities")
                fig_bars = plot_prediction_bars(all_predictions)
                st.pyplot(fig_bars)
                plt.close()
                
                # Audio visualizations
                st.subheader("🎼 Audio Visualizations")
                
                # Tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🌊 Waveform", 
                    "📊 Spectrogram", 
                    "🎨 Mel-Spectrogram", 
                    "🔢 MFCCs"
                ])
                
                with tab1:
                    st.markdown("**Time-domain representation of the audio signal**")
                    fig_wave = plot_waveform(audio, sr)
                    st.pyplot(fig_wave)
                    plt.close()
                
                with tab2:
                    st.markdown("**Frequency-time representation of the audio**")
                    fig_spec = plot_spectrogram(audio, sr)
                    st.pyplot(fig_spec)
                    plt.close()
                
                with tab3:
                    st.markdown("**Mel-scaled frequency representation (human perception)**")
                    fig_mel = plot_mel_spectrogram(audio, sr)
                    st.pyplot(fig_mel)
                    plt.close()
                
                with tab4:
                    st.markdown("**Mel-Frequency Cepstral Coefficients used for prediction**")
                    # Show MFCC of first segment
                    fig_mfcc = plot_mfcc(mfccs[0], sr)
                    st.pyplot(fig_mfcc)
                    plt.close()
                    st.caption("*Showing MFCCs from the first 3-second segment*")
                
                # Detailed results
                with st.expander("🔍 View Detailed Results"):
                    st.write("**All Predictions (per segment):**")
                    for genre, prob in zip(GENRES, all_predictions):
                        st.write(f"- {genre.capitalize()}: {prob*100:.2f}%")
                    
                    st.write(f"\n**MFCC Shape:** {mfccs.shape}")
                    st.write(f"**Audio Duration:** {len(audio)/sr:.2f} seconds")
                    st.write(f"**Sampling Rate:** {sr} Hz")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    else:
        # Instructions when no file is uploaded
        st.info("👈 Please upload an MP3 file from the sidebar to begin analysis.")
        
        st.subheader("🎯 How to Use")
        st.markdown("""
        1. **Upload** an MP3 file using the sidebar
        2. **Wait** for the model to analyze the audio
        3. **View** the predicted genre and confidence score
        4. **Explore** various audio visualizations in the tabs
        
        The model analyzes the audio by:
        - Splitting it into 10 segments (3 seconds each)
        - Extracting 13 MFCC features from each segment
        - Making predictions for each segment
        - Averaging the predictions for final result
        """)
        
        st.subheader("📝 Example Genres")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌌 Ambient**")
            st.write("- Atmospheric")
            st.write("- Slow tempo")
            st.write("- Minimal rhythm")
        
        with col2:
            st.markdown("**🎤 Pop**")
            st.write("- Catchy melodies")
            st.write("- Vocal-focused")
            st.write("- Upbeat tempo")
        
        with col3:
            st.markdown("**🎸 Rock**")
            st.write("- Electric guitars")
            st.write("- Strong beat")
            st.write("- Energetic")


if __name__ == "__main__":
    main()
