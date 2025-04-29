from flask import Flask, request, redirect, url_for, send_from_directory,jsonify,render_template
import os
from dub_video import transcribe,translation
import threading,uuid
from shared import tasks
from pydub import AudioSegment


app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
VIDEOS_FOLDER = 'videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEOS_FOLDER'] = VIDEOS_FOLDER


for folder in [UPLOAD_FOLDER, VIDEOS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['VIDEOS_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            task_id = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], task_id+"."+(file.filename.split('.')[-1]))
            file.save(file_path)
            tasks[task_id]={'status':10,'result':[],'videopath':None}
            thread = threading.Thread(target=transcribe,args=(file_path,task_id))

            # Start the thread
            thread.start()
            # mainprocess(file_path)
            
            return redirect(url_for('status',id=task_id,type=file.filename.split('.')[-1]))
    else:
        return render_template('upload.html')
    
@app.route('/<id>/transcriptionstatus',methods=['GET'])
def trstatus(id):
    return jsonify({'status':tasks[id]['status'],'message':tasks[id]['result']})


@app.route('/<id>/translationstatus',methods=['GET'])
def tlstatus(id):
    return jsonify({'status':tasks[id]['translation_status'],'message':tasks[id]['result']})

@app.route('/<id>',methods=['GET','POST'])
def status(id):
    task = tasks.get(id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    # if('transcript' in tasks[id]):
    #     type = request.args.get('type') 
    #     # thread = threading.Thread(target=translation,args=('uploads/'+id+'.'+type,id))
    #     # thread.start()
    #     return redirect(url_for('select',id=id,type=type))

    return render_template('status_1.html', status=task['status'], message=task['status'],id=id)
        
    # return jsonify(tasks)

@app.route('/<id>/vselect',methods=['GET','POST'])
def select(id):
    type = request.args.get('type') 
    if request.method == 'GET':
        speakers = {}
        for entry in tasks[id]['transcript']:
            spk = entry["speaker"]
            if spk not in speakers:
                speakers[spk] = entry  # Just take first sample per speaker

        speaker_clips = {}
        full_audio_path = f"{id}_vocals.wav"  # Assuming it's saved here

        full_audio = AudioSegment.from_wav(full_audio_path)

        for spk, entry in speakers.items():
            start_ms = int(entry["start"] * 1000)
            end_ms = int(entry["end"] * 1000)
            clip = full_audio[start_ms:end_ms]
            out_path = f"clips/{id}_{spk}.wav"
            clip.export(out_path, format="wav")
            speaker_clips[spk] = f"{id}_{spk}.wav"

        return render_template('assign_speakers.html', id=id, speakers=speakers.keys(), speaker_clips=speaker_clips,type=type)
    else:
        type = request.args.get('type') 
        assigned_voices = {}
        print(request.form)
        for key in request.form:
            assigned_voices[key] = request.form[key]
        tasks[id]['assigned_voices'] = assigned_voices

        tasks[id]['translation_status']=10
        thread = threading.Thread(target=translation, args=('uploads/' + id + ".mp4", id))
        thread.start()

        return redirect(url_for('translationstatus', id=id))

@app.route('/<id>/tstatus')
def translationstatus(id):
    
    task = tasks.get(id)
    if not task:
        return "Invalid ID", 404
    
    return render_template('status.html', status=task['translation_status'], message=task['result'],id=id)


@app.route('/end')
def end_page():
    id = request.args.get('id')
    video_path = f"videos/{id}.mp4"  # Adjust as needed
    return render_template("end.html", video_url=video_path)



@app.route('/clips/<filename>')
def serve_clip(filename):
    return send_from_directory('clips', filename)

if __name__ == '__main__':
    app.run(debug=True)