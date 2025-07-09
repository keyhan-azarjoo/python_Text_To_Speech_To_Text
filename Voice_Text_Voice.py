import whisper
import numpy as np
import resampy
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
from langdetect import detect, DetectorFactory
import pyttsx3
import io

# def speech_to_text(file_path):
#     model = whisper.load_model("base")
    
#     # Load audio with soundfile (wav/flac)
#     audio, sr = sf.read(file_path)
#     # Convert stereo to mono if needed
#     if len(audio.shape) > 1:
#         audio = np.mean(audio, axis=1)
    
#     # Convert audio to float32 (was float64 by default)
#     audio = audio.astype(np.float32)
    
#     # Resample if not 16000 Hz (Whisper expects 16kHz)
#     if sr != 16000:
#         import resampy
#         audio = resampy.resample(audio, sr, 16000)
    
#     # Pad/trim audio to fit model
#     audio = whisper.pad_or_trim(audio)
    
#     # Make log-Mel spectrogram
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
#     # Detect language (optional)
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")
    
#     # Decode
#     options = whisper.DecodingOptions()
#     result = whisper.decode(model, mel, options)
#     print("Transcription:", result.text)
#     return result.text
#transcribe_audio_file_no_ffmpeg("C:\code\Training\Python\Voice\Both_TextAndVoiceConvertor/harvard.wav")



def speech_to_text(audio_data, sr, priority_languages=["fa", "en"]):
    model = whisper.load_model("base")

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Convert to float32 if needed
    audio_data = audio_data.astype(np.float32)

    # Resample to 16000 Hz if needed
    if sr != 16000:
        audio_data = resampy.resample(audio_data, sr, 16000)

    # Pad/trim and convert to mel spectrogram
    audio_data = whisper.pad_or_trim(audio_data)
    mel = whisper.log_mel_spectrogram(audio_data).to(model.device)

    # Detect language
    _, probs = model.detect_language(mel)

    # Prioritize languages (Persian and English first)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    for lang in priority_languages:
        if lang in probs:
            top_lang = lang
            break
    else:
        top_lang = sorted_probs[0][0]  # fallback to top detected

    print(f"Detected language (biased): {top_lang}")

    # Transcribe with forced language
    options = whisper.DecodingOptions(language=top_lang)
    result = whisper.decode(model, mel, options)

    print("Transcription:", result.text)
    return result.text



# #file_path = r"C:\code\Training\Python\Voice\Both_TextAndVoiceConvertor\english_output.mp3"
# file_path = r"C:\code\Training\Python\Voice\Both_TextAndVoiceConvertor\Farsi_Female_Ghaemizade.mp3"
# audio_data, sr = sf.read(file_path)
# # Pass the loaded audio to the function
# transcription = speech_to_text(audio_data, sr)






DetectorFactory.seed = 0  # Consistent language detection
def text_to_speech(text):
    lang = detect(text)
    print(f"Detected language: {lang}")

    try:
        mp3_io = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_io)
        mp3_io.seek(0)
        return mp3_io.read()
    except Exception as e:
        print(f"gTTS failed: {e}")
        return None

    # Optional: Save MP3 to file (commented out)
    # with open(mp3_path, "wb") as f:
    #     f.write(mp3_io.getvalue())

    return mp3_io.read()


# result = text_to_speech("Hello! This is a text to speech test.")
# #text_to_speech("你好，今天天气怎么样？")
# #result = text_to_speech("سلام! حال شما چطور است؟")
# if result:
#     with open(r"C:\code\Training\Python\Voice\Both_TextAndVoiceConvertor\persian_output.mp3", "wb") as f:
#         f.write(result)
#     print("MP3 file written.")