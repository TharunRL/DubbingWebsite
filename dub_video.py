import os
import shutil
import json
import time
import requests
from io import BytesIO
import httpx
from google import genai
import subprocess
from shared import tasks
import glob
class NoVerifyClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs["verify"] = False
        super().__init__(*args, **kwargs)

httpx.Client = NoVerifyClient
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from pydub import AudioSegment

from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError

from google.cloud import texttospeech

WAIT_TIME = 5

def transcribe(x,id):
    
    def extract_audio(video_path, audio_output):
        tasks[id]['status']=20
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output)
        print(f"[INFO] Audio extracted and saved as {audio_output}")
        print(tasks[id],"\n\n\n\n\n\n")
        tasks[id]['result'].append(f"[INFO] Audio extracted and saved as {audio_output}")

    def separate_audio(input_audio):
        tasks[id]['status']=30
        command = ["python3", "-m", "demucs", "--two-stems=vocals", input_audio]
        subprocess.run(command, check=True)
        # base_name = os.path.splitext(os.path.basename(input_audio))[0]
        output_folder = f"separated/htdemucs/{id}"
        
        vocals_path = os.path.join(output_folder, "vocals.wav")
        background_path = os.path.join(output_folder, "no_vocals.wav")
        
        vocals_output = f"{id}_vocals.wav"
        background_output = f"{id}_background.wav"
        
        shutil.move(vocals_path, vocals_output)
        shutil.move(background_path, background_output)
        
        print()
        tasks[id]['result'].append("[INFO] Audio separation complete: vocals and background extracted.")
        return vocals_output, background_output
    

    def transcribe_audio(audio_file):
        tasks[id]['status']=40
        API_KEY = "ZAbKr5IyjJfZbMoW5vcNqMIP1iIimzds" 
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

        with BatchClient(settings) as client:
            try:
                job_id = client.submit_job(
                    audio=audio_file,
                    transcription_config=conf,
                )
                print(f"[INFO] Submitted transcription job {job_id}. Waiting for completion...")
                tasks[id]['result'].append(f"[INFO] Submitted transcription job {job_id}. Waiting for completion...")
                transcripts = client.wait_for_completion(job_id, transcription_format='json-v2')
                print("[INFO] Transcription complete.")
                tasks[id]['result'].append("[INFO] Transcription complete.")
            except HTTPStatusError as e:
                if e.response.status_code == 401:
                    print("[ERROR] Invalid Speechmatics API key. Please check your credentials.")
                elif e.response.status_code == 400:
                    print(f"[ERROR] {e.response.json()['detail']}")
                else:
                    raise e
        tasks[id]['status']=50
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
        tasks[id]['status']=70
        if current_block:
            transcribe.append(current_block)
        tasks[id]['transcript']=transcribe
        tasks[id]['status']=90
        return transcribe
    #split



    def process_video(input_video, output_video):

        audio_file = f"{id}.wav"
        extract_audio(input_video, audio_file)
        print(f"[INFO] Waiting {WAIT_TIME} seconds for resource stabilization...")
        time.sleep(WAIT_TIME)

        vocals_audio, background_audio = separate_audio(audio_file)
        print(f"[INFO] Waiting {WAIT_TIME} seconds for resource stabilization...")
        time.sleep(WAIT_TIME)

        transcript = transcribe_audio(vocals_audio)
        print(f"[INFO] Waiting {WAIT_TIME} seconds for resource stabilization...")
        time.sleep(WAIT_TIME)
        tasks[id]['status']=100
        #split
        
    return process_video(x, id+".mp4")

def translation(x,id):
    tasks[id]['result']=[]
    def translate_and_refine(transcript):
        json_data = json.dumps(transcript, indent=4)
        tasks[id]['translation_status']=20
        rapidapi_key = "7f3b55e0bemshb11fc9a6e4bc0a7p14b384jsn531699aebdd8"  # <-- Replace with your RapidAPI key
        
        def translate_with_rapidapi(text):
            url = "https://google-translate113.p.rapidapi.com/api/v1/translator/json"
            payload = {
                "from": "auto",
                "to": "ta",
                "common_protected_paths": ["speaker","voice"],
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
        tasks[id]['translation_status']=30
        client = genai.Client(api_key="AIzaSyBu_ggFYGfWooB7DFkyb6AF-CVKaahZALs")

        def refine_with_slang(translated_text):
            response = client.models.generate_content(
            model="gemini-2.0-flash", contents=f"{translated_text} from the above list of dictionaries i want you to convert formal Tamil text into Chennai slang used by the youth in 2020's,punctuate so that later i can use tts sound more natural with the punctuations,keep the text in length to fit the timestamp (start:end) when converted to voice , I want you to only return the final list of disctionaries in a single line."
            )
            print(response)
            tasks[id]['result'].append(response.text.split("```")[1][4:])
            return(response.text.split("```")[1][4:])
    

        i=0
        l=len(formal_tamil)
        slang_tamil=[]
        while(i<l):
            x = refine_with_slang(formal_tamil[i:l if i+10>l else i+10])
            slang_tamil.extend(json.loads(x))
            print(slang_tamil[i:])
            i+=10
        tasks[id]['translation_status']=40
        for entry in slang_tamil:
            speaker = entry["speaker"]
            entry["voice"] = tasks[id]['assigned_voices'].get(speaker, 'ta-IN-Standard-A')
        tasks[id]['result'].append("[INFO] Translation and refinement to Tamil complete.")
        print("[INFO] Translation and refinement to Tamil complete.")
        tasks[id]['translation_status']=50
        return slang_tamil

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

    default_voices = [
    'ta-IN-Standard-A',
    'ta-IN-Standard-B',
    'ta-IN-Standard-C',
    'ta-IN-Standard-D',
    'ta-IN-Wavenet-A',
    'ta-IN-Wavenet-B',
    'ta-IN-Wavenet-C',
    'ta-IN-Wavenet-D'
]


    def text_to_speech(text, voice_name):
        client = texttospeech.TextToSpeechClient()

        # voice_name = default_voices[0]
        voice = texttospeech.VoiceSelectionParams(
            language_code="ta-IN",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.2,
            pitch=1.5
        )
        
        limit = 5000
        if len(text.encode('utf-8')) > limit:
            chunks = split_text(text, limit)
        else:
            chunks = [text]
        
        combined_audio = AudioSegment.empty()
        for chunk in chunks:
            synthesis_input = texttospeech.SynthesisInput(text=chunk)
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            audio_segment = AudioSegment.from_file(BytesIO(response.audio_content), format="mp3")
            combined_audio += audio_segment
        return combined_audio

    def create_new_audio(text_array):
        text_array = sorted(text_array, key=lambda x: x['start'])
        total_duration_ms = int(max(entry['end'] for entry in text_array) * 1000)
        tasks[id]['translation_status']=60
        new_audio = AudioSegment.silent(duration=total_duration_ms)
        for entry in text_array:
            start_ms = int(entry['start'] * 1000)
            tts_audio = text_to_speech(entry['text'], entry['voice'])
            new_audio = new_audio.overlay(tts_audio, position=start_ms)
        tasks[id]['translation_status']=80
        return new_audio

    def overlay_audio_on_video(input_video, background_audio, tts_audio, output_video):
        print(input_video)
        video_clip = VideoFileClip(input_video)
        audio_clip1 = AudioFileClip(background_audio)
        tasks[id]['translation_status']=90
        temp_tts_file = "temp_tts.mp3"
        tts_audio.export(temp_tts_file, format="mp3")
        audio_clip2 = AudioFileClip(temp_tts_file)
        
        combined_audio = CompositeAudioClip([audio_clip1, audio_clip2])
        video_clip.audio = combined_audio
        video_clip.write_videofile("videos/"+output_video, codec="libx264", audio_codec="aac", fps=60)
        
        os.remove(temp_tts_file)
        tasks[id]['result'].append("[INFO] Video is dubbed.")
        print(f"[INFO] Final dubbed video saved as {output_video}")
        tasks[id]['translation_status']=100
    
    def process_video(input_video, output_video):
        refined_transcript = translate_and_refine(tasks[id]['transcript'])
        print(f"[INFO] Waiting {WAIT_TIME} seconds for resource stabilization...")
        time.sleep(WAIT_TIME)

        tts_audio_segment = create_new_audio(refined_transcript)
        print(f"[INFO] Waiting {WAIT_TIME} seconds for resource stabilization...")
        time.sleep(WAIT_TIME)

        overlay_audio_on_video(input_video, f"{id}_background.wav", tts_audio_segment, output_video)
        tasks[id]['videopath']=output_video

        def cleanup_temp_files(id):
            try:

                for ext in ["_background.wav", "_vocals.wav", ".wav"]:
                    file = f"{id}{ext}"
                    if os.path.exists(file):
                        os.remove(file)

                for file in glob.glob(f"uploads/{id}.*"):
                    os.remove(file)

                sep_dir = f"separated/htdemucs/{id}"
                if os.path.exists(sep_dir):
                    shutil.rmtree(sep_dir)

                for file in glob.glob(f"clips/{id}_*.wav"):
                    os.remove(file)
                print("[INFO] Temporary files deleted successfully.")

            except Exception as e:
                tasks[id]['result'].append(f"[ERROR] Cleanup failed: {e}")
                print(f"[ERROR] Cleanup failed: {e}")
        cleanup_temp_files(id)
        return output_video
    return process_video(x, id+".mp4")
    

        

