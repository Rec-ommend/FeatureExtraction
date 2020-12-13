from flask import Flask, jsonify, request
import timbral_models
import pandas as pd
#from gcloud import storage
import os
import shutil
import tempfile
import librosa
from spleeter.separator import *
import soundfile 

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

def timbre_similarity(timbre1, timbre2):
    similarity = 0
    for attr in attrs:
        similarity += (timbre1[attr] - timbre2[attr]) ** 2
    return similarity


def timbre_similarity_list(input_vocal_path: str):
    timbre = timbral_models.timbral_extractor(input_vocal_path)
    database_path = 'gs://timbre_an/crawling.csv'
    dataframe = pd.read_csv(database_path)

    dataframe['timbre_similarity'] = 0

    for index in range(dataframe.shape[0]):
        dataframe['timbre_similarity'][index] = timbre_similarity(
            timbre, dict(dataframe[attrs].loc[index]))

    new_df = dataframe.sort_values(by=['timbre_similarity'])
    return new_df[['id', 'title', 'singer', 'genre', 'release', 'timbre_similarity']]


@app.route('/voice', methods=['POST'])
def voice():
    reqfile = request.files['voice']
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
        result = timbre_similarity_list(new_vocal_path)
        os.remove(temp.name)
        os.remove(new_vocal_path)
        print(result)
        return result


if __name__ == '__main__':
    app.run()
