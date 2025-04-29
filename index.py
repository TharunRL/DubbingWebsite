#!/usr/bin/env python3
import os
import json
import time
import requests
import subprocess
from io import BytesIO
import numpy as np
import httpx
from google import genai
from concurrent.futures import ThreadPoolExecutor

# Force httpx to not verify certificates (if needed)
class NoVerifyClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs["verify"] = False
        super().__init__(*args, **kwargs)
httpx.Client = NoVerifyClient

# Video and audio processing libraries
from moviepy import VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
from pydub import AudioSegment

# Speechmatics modules for transcription
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError

# Google Cloud Text-to-Speech client
from google.cloud import texttospeech

# Set a wait time (set to 0 if not needed)
WAIT_TIME = 0

#############################################
# 1. In-memory audio extraction from video
#############################################
def extract_audio_from_video(video_path):
    audio = AudioSegment.from_file(video_path, format="mp4")
    print("[INFO] Audio extracted from video (in memory).")
    return audio

####################################################
# 2. In-memory Demucs separation with GPU support
####################################################
def separate_audio_in_memory(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels)).T
    else:
        samples = samples[np.newaxis, :]
    norm = float(1 << (8 * audio_segment.sample_width - 1))
    samples = samples.astype(np.float32) / norm

    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs").to(device)
    waveform = torch.tensor(samples).unsqueeze(0).to(device)  # shape: (1, channels, samples)

    with torch.no_grad():
        estimates = apply_model(model, waveform, overlap=0.25, progress=False)
    estimates = estimates[0]  # remove batch dimension

    # Assume index 0: vocals, index 1: accompaniment
    vocals_np = estimates[:, 0, :].cpu().numpy()
    accomp_np = estimates[:, 1, :].cpu().numpy()

    vocals_np = (vocals_np * norm).astype(np.int16)
    accomp_np = (accomp_np * norm).astype(np.int16)

    def to_audiosegment(arr):
        if arr.shape[0] > 1:
            interleaved = arr.T.flatten().tobytes()
        else:
            interleaved = arr.flatten().tobytes()
        return AudioSegment(
            data=interleaved,
            sample_width=audio_segment.sample_width,
            frame_rate=audio_segment.frame_rate,
            channels=audio_segment.channels
        )
    vocals_audio = to_audiosegment(vocals_np)
    background_audio = to_audiosegment(accomp_np)

    print("[INFO] Audio separation complete (in memory).")
    return vocals_audio, background_audio

####################################################
# 3. Transcription (in-memory via Speechmatics)
####################################################
def transcribe_audio(audio_segment):
    """
    Transcribes the provided vocals AudioSegment using the Speechmatics API.
    Exports the AudioSegment to an in-memory buffer and passes a tuple of (filename, bytes).
    """
    API_KEY = "t7aES4wtwT1soOx0zuT8JNnw1QzKXiF8"  # Replace with your Speechmatics API key
    LANGUAGE = "en"
    settings = ConnectionSettings(
        url="https://asr.api.speechmatics.com/v2",
        auth_token=API_KEY,
    )
    conf = {
        "type": "transcription",
        "transcription_config": {
            "language": LANGUAGE,
            "diarization": "speaker",
            "speaker_diarization_config": {
                "speaker_sensitivity": 0.8,
            }
        },
    }
    # Export audio to an in-memory buffer in WAV format
    buffer = BytesIO()
    audio_segment.export(buffer, format="wav")
    audio_bytes = buffer.getvalue()

    with BatchClient(settings) as client:
        try:
            # Pass the audio as a tuple: (filename, bytes)
            job_id = client.submit_job(audio=("audio.wav", audio_bytes), transcription_config=conf)
            print(f"[INFO] Submitted transcription job {job_id}. Waiting for completion...")
            transcripts = client.wait_for_completion(job_id, transcription_format='json-v2')
            print("[INFO] Transcription complete.")
        except HTTPStatusError as e:
            if e.response.status_code == 401:
                print("[ERROR] Invalid Speechmatics API key. Please check your credentials.")
            elif e.response.status_code == 400:
                print(f"[ERROR] {e.response.json()['detail']}")
            else:
                raise e

    results = transcripts.get("results", [])
    transcribe = []
    current_block = None

    for seg in results:
        token = seg["alternatives"][0]["content"]
        speaker = seg["alternatives"][0].get("speaker", "Unknown")
        seg_start = seg["start_time"]
        seg_end = seg["end_time"]
        seg_type = seg.get("type", "word")
        attaches = seg.get("attaches_to", None)

        if current_block is None:
            current_block = {"speaker": speaker, "text": token, "start": seg_start, "end": seg_end}
        else:
            if speaker == current_block["speaker"]:
                if attaches == "previous" or seg_type == "punctuation":
                    current_block["text"] += token
                else:
                    current_block["text"] += " " + token
                current_block["end"] = seg_end
            else:
                transcribe.append(current_block)
                current_block = {"speaker": speaker, "text": token, "start": seg_start, "end": seg_end}

    if current_block:
        transcribe.append(current_block)

    return transcribe
###################################################
# 4. Translation and refinement (parallelized)
####################################################
def translate_and_refine(transcript):
    json_data = json.dumps(transcript, indent=4)
    rapidapi_key = "7f3b55e0bemshb11fc9a6e4bc0a7p14b384jsn531699aebdd8"  # Replace with your RapidAPI key

    def translate_with_rapidapi(text):
        url = "https://google-translate113.p.rapidapi.com/api/v1/translator/json"
        payload = {
            "from": "auto",
            "to": "ta",
            "common_protected_paths": ["speaker"],
            "json": text
        }
        headers = {
            "x-rapidapi-key": rapidapi_key,
            "x-rapidapi-host": "google-translate113.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        return response.json()["trans"]

    formal_tamil = translate_with_rapidapi(json_data)
    client = genai.Client(api_key="AIzaSyBu_ggFYGfWooB7DFkyb6AF-CVKaahZALs")

    def refine_with_slang(chunk):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{chunk} from the above list of dictionaries i want you to convert formal Tamil text into Chennai slang used by the youth in 2020's, punctuate so that later i can use tts sound more natural with the punctuations, keep the text short to fit the timestamp (start:end) when converted to voice, if the text doesnt make sense in continuous or if it should be said slower or in parts or with pauses split the text and put into a dictionary right next to it and change the start end time with a gap between the two. I want you to only return the final list of disctionaries in a single line"
        )
        # Extract refined JSON from the response
        return json.loads(response.text.split("```")[1][4:])

    # Break the formal_tamil into chunks (here using a fixed chunk size)
    chunk_size = 10
    chunks = [formal_tamil[i:i+chunk_size] for i in range(0, len(formal_tamil), chunk_size)]
    slang_tamil = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(refine_with_slang, chunks)
    for res in results:
        slang_tamil.extend(res)

    print("[INFO] Translation and refinement to Tamil complete.")
    return slang_tamil

####################################################
# 5. TTS synthesis (parallelized)
####################################################
default_voices = [
    'ta-IN-Standard-A',
    'ta-IN-Standard-B',
    'ta-IN-Standard-C',
    'ta-IN-Standard-D'
]

def split_text(text, limit=5000):
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        candidate = f"{current_chunk} {word}".strip() if current_chunk else word
        if len(candidate.encode('utf-8')) > limit:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = word
            else:
                while len(word.encode('utf-8')) > limit:
                    part = word[:limit]
                    chunks.append(part)
                    word = word[limit:]
                current_chunk = word
        else:
            current_chunk = candidate
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def synthesize_chunk(chunk, voice):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=chunk)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.2,
        pitch=1.5
    )
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    return AudioSegment.from_file(BytesIO(response.audio_content), format="mp3")

def text_to_speech(text, speaker):
    voice = texttospeech.VoiceSelectionParams(
        language_code="ta-IN",
        name=default_voices[0]
    )
    limit = 5000
    chunks = split_text(text, limit) if len(text.encode('utf-8')) > limit else [text]
    
    # Use ThreadPoolExecutor to synthesize chunks concurrently
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda chunk: synthesize_chunk(chunk, voice), chunks))
    combined_audio = AudioSegment.empty()
    for seg in results:
        combined_audio += seg
    return combined_audio

def create_new_audio(text_array):
    text_array = sorted(text_array, key=lambda x: x['start'])
    total_duration_ms = int(max(entry['end'] for entry in text_array) * 1000)
    new_audio = AudioSegment.silent(duration=total_duration_ms)
    for entry in text_array:
        start_ms = int(entry['start'] * 1000)
        tts_audio = text_to_speech(entry['text'], entry['speaker'])
        new_audio = new_audio.overlay(tts_audio, position=start_ms)
    return new_audio

####################################################
# 6. Conversion for MoviePy and final video output
####################################################
def audiosegment_to_np(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
    else:
        samples = samples[:, np.newaxis]
    max_val = float(1 << (8 * audio_segment.sample_width - 1))
    np_audio = samples.astype(np.float32) / max_val
    return np_audio, audio_segment.frame_rate

def overlay_audio_on_video(input_video, background_audio, tts_audio, output_video):
    final_audio = background_audio.overlay(tts_audio)
    np_audio, fps = audiosegment_to_np(final_audio)
    audio_clip = AudioArrayClip(np_audio, fps=fps)
    video_clip = VideoFileClip(input_video)
    video_clip.audio = audio_clip
    video_clip.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=60)
    print(f"[INFO] Final dubbed video saved as {output_video}")

####################################################
# 7. Full processing pipeline (optimized)
####################################################
def process_video(input_video, output_video):
    full_audio = extract_audio_from_video(input_video)
    if WAIT_TIME:
        time.sleep(WAIT_TIME)
    
    vocals_audio, background_audio = separate_audio_in_memory(full_audio)
    if WAIT_TIME:
        time.sleep(WAIT_TIME)
    
    transcript = transcribe_audio(vocals_audio)
    if WAIT_TIME:
        time.sleep(WAIT_TIME)
    
    refined_transcript = translate_and_refine(transcript)
    if WAIT_TIME:
        time.sleep(WAIT_TIME)
    
    tts_audio_segment = create_new_audio(refined_transcript)
    if WAIT_TIME:
        time.sleep(WAIT_TIME)
    
    overlay_audio_on_video(input_video, background_audio, tts_audio_segment, output_video)

def main():
    process_video("Interstellar_2014_720_Trim.mp4", "final_dubbed.mp4")

if __name__ == '__main__':
    main()
