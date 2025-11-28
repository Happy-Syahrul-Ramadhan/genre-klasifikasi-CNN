# 🚀 Streamlit Production Setup - Quick Guide

## ✅ What Was Fixed

### 1. **requirements.txt** - Package Version Compatibility
```diff
- tensorflow>=2.16.0   ❌ (incompatible with NumPy)
- tensorflow==2.15.0   ❌ (not available for Python 3.13)
+ tensorflow==2.20.0   ✅ (latest, Python 3.13 compatible)

- streamlit==1.31.0    ❌ (requires protobuf<5, conflicts with TF 2.20)
+ streamlit==1.40.2    ✅ (supports protobuf 5.x)

- numpy<2.0.0          ❌ (too broad)
+ numpy>=1.26.0,<2.0.0 ✅ (compatible with TF 2.20 and Python 3.13)

- scipy==1.11.4        ❌ (tries to build from source, needs gfortran)
+ scipy==1.14.1        ✅ (pre-built wheels available)

+ soundfile==0.12.1    ✅ (missing librosa dependency)
- protobuf==4.25.5     ❌ (conflicts: Streamlit needs <5, TensorFlow needs >=5.28)
+ (auto-resolved)      ✅ (pip resolves compatible protobuf version)
```

**Critical Issues Fixed:**
1. **Python 3.13 Compatibility**: Streamlit Cloud uses Python 3.13. TensorFlow 2.20.0 is the first version that supports it.
2. **Protobuf Conflict**: TensorFlow 2.20.0 requires `protobuf>=5.28.0`, but Streamlit 1.31.0 requires `protobuf<5`. Solution: Upgrade Streamlit to 1.40.2 which supports protobuf 5.x.

### 2. **packages.txt** - System Dependencies
```
libsndfile1   → Required by librosa for audio file reading
ffmpeg        → Audio codec support for MP3/WAV files
gfortran      → Fortran compiler for SciPy
libblas-dev   → Linear algebra library for SciPy
liblapack-dev → Linear algebra library for SciPy
```

### 3. **app.py** - Fixed Model Path
```python
# BEFORE (doesn't work on Streamlit Cloud):
MODEL_PATH = r'D:/Music-Genre-Classification-.../models/model_cnn2.h5'

# AFTER (works everywhere):
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_cnn2.h5')
```

### 4. **.gitattributes** - Git LFS for Large Files
```
*.h5 filter=lfs diff=lfs merge=lfs -text
```

### 5. **config.toml** - Production Settings
```toml
[server]
headless = true         # No GUI on server
enableCORS = false      # Security
port = 8501            # Standard Streamlit port
```

---

## 🎯 Deploy Now!

### Step 1: Verify Changes Are Pushed
```bash
git log --oneline -5
# Should show commits with deployment fixes
```

### Step 2: Go to Streamlit Cloud
1. Visit: https://share.streamlit.io
2. Click **"New app"**
3. Select your repository: `Happy-Syahrul-Ramadhan/genre_music_classification_with_CNN`
4. Set:
   - **Branch**: `main`
   - **Main file path**: `streamlit/app.py`
5. Click **"Deploy"**

### Step 3: Wait for Build
- Build time: ~3-5 minutes
- Watch logs for errors
- If successful, you'll get a public URL: `https://[your-app-name].streamlit.app`

---

## 🐛 If Deployment Still Fails

### Error: "installer returned a non-zero exit code"
**Root Causes Fixed:**
1. ✅ TensorFlow 2.15.0 not available for Python 3.13 → Upgraded to TensorFlow 2.20.0
2. ✅ SciPy 1.11.4 requires Fortran compiler → Upgraded to SciPy 1.14.1 (pre-built wheels)
3. ✅ Protobuf conflict: Streamlit 1.31.0 requires `<5`, TensorFlow 2.20.0 requires `>=5.28.0` → Upgraded Streamlit to 1.40.2

**Additional Check:**
```bash
# Test locally first (if you have Python 3.13)
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

### Error: "Cannot install... conflicting dependencies (protobuf)"
**Example Error:**
```
streamlit 1.31.0 depends on protobuf<5
tensorflow 2.20.0 depends on protobuf>=5.28.0
```

**Current Fix Applied:** ✅ Upgraded Streamlit to 1.40.2 which supports protobuf 5.x

**Manual Fix (if needed):**
```bash
# Update requirements.txt
streamlit==1.40.2  # or newer
tensorflow==2.20.0
# Remove explicit protobuf line - let pip resolve it
```

### Error: "Model file not found"
**Current Fix Applied:** ✅ Model path uses relative `os.path.join()`

**Verify Model Exists:**
```bash
ls -lh models/model_cnn2.h5
# Should show file size (~1-2 MB)
```

### Error: "File size exceeds 100MB"
**Current Fix Applied:** ✅ Git LFS configured in `.gitattributes`

**If Model > 100MB:**
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.h5"

# Verify tracking
git lfs ls-files

# Commit and push
git add .gitattributes models/*.h5
git commit -m "Use Git LFS for model files"
git push origin main
```

---

## 📊 Expected Deployment Result

### ✅ Successful Deployment Indicators:
- App builds without errors (logs show "Your app is live!")
- Upload button works
- Sample MP3 prediction returns genre
- All visualizations render
- No memory/timeout errors

### 🎵 Test Your Deployed App:
1. Upload a 30-second MP3 file
2. Wait for processing (~10 seconds)
3. Check predictions for 10 segments
4. Verify genre classification (Ambient/Pop/Rock)

---

## 📝 Post-Deployment Checklist

- [ ] App loads at public URL
- [ ] No errors in deployment logs
- [ ] File upload accepts MP3 files
- [ ] Model prediction works (shows genre + confidence)
- [ ] Waveform visualization renders
- [ ] Spectrogram displays correctly
- [ ] MFCC features show properly
- [ ] All 10 segments are processed

---

## 🔗 Resources

- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Git LFS Guide**: https://git-lfs.github.com/
- **Your Repo**: https://github.com/Happy-Syahrul-Ramadhan/genre_music_classification_with_CNN

---

## 💡 Pro Tips

1. **First deployment takes longest** (5-10 min) - subsequent updates are faster
2. **Free tier limits**: 1 GB RAM, 1 CPU core, 50 GB bandwidth/month
3. **App hibernates** after 7 days of inactivity
4. **Custom domain**: Available with paid plan

---

**Status**: ✅ Ready for deployment (Python 3.12 compatible)  
**Last Updated**: 2025-11-28  
**Commit**: `8f8546a`  
**TensorFlow Version**: 2.20.0 (latest, Python 3.12 support)

Try deploying now! 🚀

---

## 🔄 Changelog

**v2 (2025-11-28 01:41):**
- ✅ Fixed TensorFlow version: 2.15.0 → 2.20.0
- ✅ Updated NumPy range for Python 3.12 compatibility
- ✅ Removed protobuf pinning (handled by TensorFlow 2.20)

**v1 (2025-11-28):**
- Initial deployment setup with relative paths
- Added packages.txt for system dependencies
- Configured Git LFS for large files
