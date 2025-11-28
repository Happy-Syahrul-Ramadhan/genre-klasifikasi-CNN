# 🚨 Streamlit Deployment - Quick Fixes

## ✅ Current Configuration (WORKING)

```txt
# requirements.txt
streamlit==1.31.0
tensorflow==2.20.0          ← Python 3.13 compatible
librosa==0.10.1
numpy>=1.26.0,<2.0.0       ← TensorFlow 2.20 compatible
matplotlib==3.8.2
scipy==1.14.1              ← Pre-built wheels available
soundfile==0.12.1
protobuf==4.25.5           ← TensorFlow 2.20 compatible
```

```txt
# packages.txt
libsndfile1
ffmpeg
gfortran                   ← Required for SciPy
libblas-dev                ← Linear algebra for SciPy
liblapack-dev              ← Linear algebra for SciPy
```

---

## 🔍 Error Messages & Solutions

### ❌ "Could not find a version that satisfies the requirement tensorflow==2.15.0"

**Problem:** TensorFlow 2.15.0 not available for Python 3.13  
**Solution:** ✅ **ALREADY FIXED** - Using TensorFlow 2.20.0

---

### ❌ "Unknown compiler(s): gfortran" / "ERROR: metadata-generation-failed"

**Problem:** SciPy 1.11.4 tries to build from source, needs Fortran compiler  
**Solution:** ✅ **ALREADY FIXED** 
- Upgraded to SciPy 1.14.1 (has pre-built wheels)
- Added gfortran, libblas-dev, liblapack-dev to packages.txt

---

### ❌ "installer returned a non-zero exit code"

**Possible Causes:**
1. **Wrong TensorFlow version** → ✅ Fixed (using 2.20.0)
2. **NumPy incompatibility** → ✅ Fixed (using >=1.26.0,<2.0.0)
3. **SciPy needs compiler** → ✅ Fixed (using 1.14.1 + gfortran)
4. **Missing system packages** → ✅ Fixed (packages.txt exists)

---

### ❌ "Model file not found"

**Check:**
```python
# In app.py - should be relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_cnn2.h5')
```

**Verify model exists:**
```bash
ls -lh models/model_cnn2.h5
# Should show ~1-2 MB file
```

---

### ❌ "libsndfile.so.1: cannot open shared object file"

**Solution:** ✅ **ALREADY FIXED** - `packages.txt` includes `libsndfile1`

---

### ❌ "This app has gone over its resource limits"

**Causes:**
- Model file > 100 MB → Use Git LFS
- Too much memory usage → Optimize code

**Check model size:**
```bash
du -h models/model_cnn2.h5
```

---

## 🎯 Deployment Checklist

- [x] TensorFlow 2.20.0 (Python 3.12 compatible)
- [x] NumPy >=1.26.0 (TF 2.20 compatible)
- [x] Relative paths in app.py
- [x] packages.txt with system dependencies
- [x] .streamlit/config.toml configured
- [ ] Git repository pushed to GitHub
- [ ] Deploy on https://share.streamlit.io

---

## 🚀 Deploy Command

```bash
# From project root
git add .
git commit -m "Ready for Streamlit deployment"
git push origin main

# Then go to: https://share.streamlit.io
# Repository: Happy-Syahrul-Ramadhan/genre_music_classification_with_CNN
# Main file: streamlit/app.py
# Branch: main
```

---

## 📊 Expected Deployment Time

- **First deploy:** 3-5 minutes
- **Updates:** 1-2 minutes
- **Failed builds:** Check logs immediately

---

## 🔗 Resources

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **TensorFlow Compatibility:** https://www.tensorflow.org/install
- **Python 3.12 Notes:** https://docs.python.org/3.12/whatsnew/3.12.html

---

## ✅ Current Status

**Commit:** `5d5fd61`  
**Status:** 🟢 Ready to deploy  
**Last Fix:** Python 3.12 compatibility (TensorFlow 2.20.0)  
**Date:** 2025-11-28 01:45 UTC

**Deploy now at:** https://share.streamlit.io
