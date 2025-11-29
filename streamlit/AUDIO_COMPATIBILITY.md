# Audio File Compatibility Guide

## Supported Formats

The app supports multiple audio formats with varying levels of compatibility:

### ✅ **Recommended: WAV Files**
- **Best compatibility** - native support via soundfile
- No codec issues
- Fastest loading
- **Use this format if you're having issues with MP3**

### ⚠️ **MP3 Files**
- Supported but may have issues on some deployments
- Requires system codecs (libmpg123, ffmpeg)
- May show warnings but usually works
- If MP3 fails, convert to WAV

### 🔄 **Other Formats** (OGG, M4A)
- Supported via ffmpeg backend
- Compatibility depends on system installation
- May be slower to load

## Common Issues & Solutions

### Issue: "Giving up searching valid MPEG header"
**Cause:** MP3 file is corrupted or uses unsupported codec  
**Solution:**
1. Try converting to WAV format
2. Re-export the MP3 with standard settings
3. Try a different audio file

### Issue: "PySoundFile failed. Trying audioread instead"
**Cause:** Format not supported by soundfile (usually MP3)  
**Solution:** This is normal - the app will fallback to audioread. If it still fails, use WAV.

### Issue: "All audio loading methods failed"
**Cause:** File is corrupted or in unsupported format  
**Solution:**
1. Convert to WAV using: `ffmpeg -i input.mp3 output.wav`
2. Check file integrity
3. Try a different source file

## Converting Audio Files

### Using FFmpeg (Command Line)
```bash
# MP3 to WAV
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav

# Any format to WAV (optimized for app)
ffmpeg -i input.m4a -ar 22050 -ac 1 -acodec pcm_s16le output.wav
```

Parameters explained:
- `-ar 22050`: Sample rate (matches app requirement)
- `-ac 1`: Mono channel
- `-acodec pcm_s16le`: 16-bit PCM (standard WAV)

### Using Online Tools
- [CloudConvert](https://cloudconvert.com/mp3-to-wav)
- [Online Audio Converter](https://online-audio-converter.com/)

### Using Audacity (Free Software)
1. Open your audio file
2. File → Export → Export as WAV
3. Set Rate: 22050 Hz
4. Set Channels: Mono

## Technical Details

### Audio Processing Parameters
- **Sampling Rate:** 22050 Hz
- **Duration:** 30 seconds (or full track if shorter)
- **Channels:** Mono
- **Segments:** 10 × 3-second clips
- **Features:** 13 MFCCs per segment

### Backend Loading Order
1. **SoundFile** (libsndfile) - fastest, WAV/FLAC/OGG
2. **Audioread** (ffmpeg/libmpg123) - slower, MP3/M4A
3. **Raw file reading** - last resort fallback

## Deployment Notes

### Streamlit Cloud
The following system packages are installed:
- `libsndfile1` - soundfile backend
- `ffmpeg` - audio/video codecs
- `libmpg123-0` + `libmpg123-dev` - MP3 decoding
- `libavcodec-extra` - additional codecs

### If MP3 Still Fails
Some MP3 files use non-standard encoding that these codecs can't decode. **Always recommend WAV format** for production use.

## Quick Reference

| Format | Compatibility | Speed | Recommendation |
|--------|--------------|-------|----------------|
| WAV    | ✅ Excellent  | ⚡ Fast | **Use this** |
| MP3    | ⚠️ Good      | 🐢 Slow | Fallback |
| OGG    | ⚠️ Good      | 🐢 Slow | Fallback |
| M4A    | ⚠️ Fair      | 🐢 Slow | Not recommended |
| FLAC   | ✅ Excellent  | ⚡ Fast | Alternative |

## Testing Your Audio File

Before uploading to the app, you can test if your file is valid:

```python
import librosa
# If this works, your file should work in the app
audio, sr = librosa.load('your_file.mp3', sr=22050, duration=30)
print(f"Loaded {len(audio)} samples at {sr} Hz")
```

## Support

If you continue to have issues:
1. Verify file is not corrupted (play in media player)
2. Convert to WAV format
3. Check file is at least 3 seconds long
4. Try a different audio file to rule out app issues
