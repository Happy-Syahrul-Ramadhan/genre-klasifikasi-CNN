# Streamlit Cloud Deployment Checklist

## ✅ Pre-Deployment Checklist

### 1. Files Required
- [ ] `streamlit/app.py` - Main application
- [ ] `streamlit/requirements.txt` - Python dependencies (FIXED versions)
- [ ] `streamlit/packages.txt` - System dependencies (libsndfile1, ffmpeg)
- [ ] `streamlit/.streamlit/config.toml` - Streamlit configuration
- [ ] `models/model_cnn2.h5` - Pre-trained CNN model (~2 MB)

### 2. Code Compatibility
- [ ] No absolute paths (use `os.path` for relative paths)
- [ ] Model path uses `os.path.join(os.path.dirname(__file__), ...)`
- [ ] All imports are in `requirements.txt`
- [ ] TensorFlow version compatible with Streamlit Cloud (2.15.0)
- [ ] NumPy version < 2.0.0 (required for TensorFlow 2.15)

### 3. GitHub Repository
- [ ] All files committed and pushed to `main` branch
- [ ] `.gitignore` excludes large files (>100MB)
- [ ] Git LFS configured for `.h5` files (if >100MB)
- [ ] Repository is public or Streamlit Cloud has access

### 4. Dependencies Fixed
- [ ] `streamlit==1.31.0` (stable version)
- [ ] `tensorflow==2.15.0` (compatible with NumPy 1.24.x)
- [ ] `librosa==0.10.1` (audio processing)
- [ ] `numpy==1.24.3` (required for TensorFlow 2.15)
- [ ] `soundfile==0.12.1` (librosa dependency)
- [ ] `protobuf==3.20.3` (TensorFlow compatibility)

## 🚀 Deployment Steps

### Step 1: Commit Changes
```bash
git add .
git commit -m "Fix: Update requirements.txt for Streamlit Cloud compatibility"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click "New app"
3. Connect GitHub repository: `Happy-Syahrul-Ramadhan/genre_music_classification_with_CNN`
4. Set main file path: `streamlit/app.py`
5. Branch: `main`
6. Click "Deploy"

### Step 3: Monitor Deployment
- Watch deployment logs for errors
- Common errors:
  - **"installer returned non-zero exit code"** → Check `requirements.txt` versions
  - **"Model file not found"** → Check model path is relative
  - **"NumPy version incompatible"** → Ensure NumPy < 2.0.0
  - **"Memory exceeded"** → Model file too large (use Git LFS)

## 🔧 Troubleshooting

### Error: "installer returned a non-zero exit code"
**Cause:** Incompatible package versions
**Fix:** Use EXACT versions in `requirements.txt`:
```
streamlit==1.31.0
tensorflow==2.15.0
numpy==1.24.3
```

### Error: "Model file not found"
**Cause:** Absolute path in code
**Fix:** Update `app.py`:
```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_cnn2.h5')
```

### Error: "libsndfile.so.1: cannot open shared object file"
**Cause:** Missing system libraries
**Fix:** Create `packages.txt`:
```
libsndfile1
ffmpeg
```

### Error: "This app has gone over its resource limits"
**Cause:** Model file >100MB or too much memory usage
**Fix:** 
1. Use Git LFS for model files
2. Optimize model (reduce parameters)
3. Use lighter TensorFlow build

## 📊 Post-Deployment Verification

- [ ] App loads without errors
- [ ] File upload works (test with MP3 file)
- [ ] Model prediction returns results
- [ ] Visualizations render correctly
- [ ] All 3 genres (Ambient, Pop, Rock) can be predicted
- [ ] App doesn't crash after multiple predictions

## 🔒 Security Notes

- Model file should be committed to repository (or use Git LFS)
- No API keys or secrets in code
- Use Streamlit secrets for sensitive data (if needed)

## 📝 Current Status

**Last Updated:** 2025-11-28
**Status:** Ready for deployment
**Issues Fixed:**
- ✅ Updated `requirements.txt` with compatible versions
- ✅ Fixed model path to use relative paths
- ✅ Added `packages.txt` for system dependencies
- ✅ Updated Streamlit config for production
- ✅ Added Git LFS configuration for large files
