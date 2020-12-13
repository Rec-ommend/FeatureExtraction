from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Flask, jsonify, request
import timbral_models
import pandas as pd
import numpy as np
#from gcloud import storage
import os
import shutil
import tempfile
import librosa
from spleeter.separator import *
import soundfile 
import subprocess
app = Flask(__name__)
CORS(app)

class ov_Seperator(Separator):
    def save_to_file(
            self,
            sources,
            audio_descriptor,
            destination,
            filename_format='{filename}/{instrument}.{codec}',
            codec='wav',
            audio_adapter=get_default_audio_adapter(),
            bitrate='128k',
            synchronous=True):
        filename_format = '{filename}.{codec}'

        foldername = basename(dirname(audio_descriptor))
        filename = splitext(basename(audio_descriptor))[0]
        generated = []
        data = sources['vocals']
        print(filename, foldername, destination)
        path = os.path.join(destination, filename_format.format(
            filename='_'+filename,
            foldername=foldername,
            codec=codec,
        ))
        directory = os.path.dirname(path)
        print(path)
        generated.append(path)
        if self._pool:
            task = self._pool.apply_async(audio_adapter.save, (
                path,
                data,
                self._sample_rate,
                codec,
                bitrate))
            self._tasks.append(task)
        else:
            audio_adapter.save(
                path,
                data,
                self._sample_rate,
                codec,
                bitrate)
        if synchronous and self._pool:
            self.join()


app = Flask(__name__)

attrs = ['hardness', 'depth', 'brightness',
         'roughness', 'warmth', 'sharpness', 'boominess']
attrval = {'hardness':[43.21439351,59.75435847],'depth':[24.29675547,58.40741266],'brightness':[52.26069264,71.37626087],'roughness':[24.66124888,57.98019528],'warmth':[33.21028496,54.25769172],'sharpness':[37.02482355,60.0009997],'boominess':[2.078721208,41.60153737]}

separator = ov_Seperator('spleeter:2stems', multiprocess=False)


def vocal_extract(input_path, codec="mp3", bitrate='40k'):
    head, tail = os.path.split(input_path.name)
    separator.separate_to_file(
        input_path.name, head, codec=codec, bitrate=bitrate)
    head, tail = os.path.split(input_path.name)
    print(head,tail)
    newpath = os.path.join(head,'_'+tail)
    filename, file_extension = os.path.splitext(newpath)
    new_vocal_path = filename+'.wav'
    new_vocal_sample_rate = 22050
    y, sr = librosa.load(input_path.name, new_vocal_sample_rate)
    soundfile.write(new_vocal_path,y,sr)
    return new_vocal_path

def convert_and_split(input_path, return_path):
    command = ['ffmpeg', '-i',input_path.name, return_path]
    subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)

def song_extract(input_path):
    head, tail = os.path.split(input_path.name)
    print(head,tail)
    newpath = os.path.join(head,'_'+tail)
    filename, file_extension = os.path.splitext(newpath)
    return_path = filename+'.wav'
    convert_and_split(input_path,return_path)
    return return_path

def timbre_similarity(timbre1, timbre2):
    similarity = 0
    for attr in attrs:
        if not np.isnan(pd.to_numeric(timbre2[attr],errors='coerce')):
            similarity += (timbre1[attr] - pd.to_numeric(timbre2[attr])) ** 2
    return similarity


def timbre_similarity_list(input_vocal_path,genre,start,end,song = False):
    timbre = timbral_models.timbral_extractor(input_vocal_path)
    or_timbre = timbre.copy()
    if song:
        database_path = 'gs://timbre_an/crawling2.csv'
    else:
        database_path = 'gs://timbre_an/crawling.csv'
    dataframe = pd.read_csv(database_path)
    dataframe['timbre_similarity'] = 0
    for index in range(dataframe.shape[0]):
        dataframe['timbre_similarity'][index] = timbre_similarity(
            timbre, dict(dataframe[attrs].loc[index]))
    for a in attrs:
        minv = attrval[a][0]
        maxv = attrval[a][1]
        timbre[a] = (timbre[a] - minv) / (maxv - minv)
    new_df = dataframe.sort_values(by=['timbre_similarity'])
    new_df = new_df[new_df.timbre_similarity!=0]
    new_df=new_df.replace(np.nan, '1')
    print(new_df.release,new_df.genre)
    masked =False
    if type(genre) == list:
        mask = None
        for singlegen in genre:
            #if new_df.genre.str.contains(singlegen, na=False) !=None:
            if mask is None:
                mask = new_df.genre.str.contains(singlegen, na=False)
                masked = True
            else:
                mask = mask | new_df.genre.str.contains(singlegen, na=False)
        if start !='0' or end !='9':
            if mask is None:
                mask = start< new_df.release & new_df.release<end
            else:
                mask = mask & (start<new_df.release) & (new_df.release<end)
            masked = True
    else:
        mask = False
        if genre:
            mask = new_df.genre.str.contains(genre, na=False)
            masked =True
        if start !='0' or end !='9':
            if genre:
                mask = mask & ((start< new_df.release) & (new_df.release<end))
            else:
                mask = ((start< new_df.release) & (new_df.release<end))
            masked = True
    if masked:
        res_df = new_df.loc[mask,:]
    else:
        res_df = new_df
    print(res_df)

    return or_timbre,timbre, res_df[['id', 'title', 'singer', 'genre', 'release', 'timbre_similarity']].head(10)


@app.route('/voice', methods=['POST'])
def voice():
    reqfile = request.files['voice']
    genre = request.form.getlist('genre')
    start = request.form.get('start','0')
    end = request.form.get('end','9')
    print(request.files['voice'])
    filename, file_extension = os.path.splitext(reqfile.filename)

    temp = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)

    shutil.copyfileobj(reqfile, temp)
    temp.close()
    try:
        print(temp.name)
    #        shutil.copy2(temp.name,'C:/Users/small/hi.mp3')
        #os.system("start "+temp.name)
        new_vocal_path = vocal_extract(temp)
    finally:
        or_timbre, timbre, result = timbre_similarity_list(new_vocal_path,genre,start,end)
        os.remove(temp.name)
        os.remove(new_vocal_path)
        print(result)
        result = result.to_json(orient="records")
        return {'song':result,'timbre':timbre,'origin':or_timbre}

@app.route('/song',methods=['POST'])
def song():
    reqfile = request.files['voice']
    genre = request.form.getlist('genre')
    start = request.form.get('start','0')
    end = request.form.get('end','9')
    youtube = request.form.get('youtube','')
    filename, file_extension = os.path.splitext(reqfile.filename)

    temp = tempfile.NamedTemporaryFile(suffix=file_extension, delete = False)

    shutil.copyfileobj(reqfile, temp)
    temp.close()
    try:
        new_vocal_path = song_extract(temp)
    finally:
        or_timbre,timbre,result = timbre_similarity_list(new_vocal_path, genre, start, end,song=True)
        os.remove(temp.name)
        os.remove(new_vocal_path)
        result = result.to_json(orient="records")
        return {'song':result, 'timbre':timbre,'origin':or_timbre}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
