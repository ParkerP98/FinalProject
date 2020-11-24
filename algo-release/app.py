import base64
import numpy as np
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
import librosa
import json
import os
from scipy.io import wavfile
from scipy import stats
import math
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)

    wav_file = open("a.wav", "wb")
    encoded = message['audio']
    e = encoded.split(',')
    e = e[1]
    decode_string = base64.b64decode(e)
    wav_file.write(decode_string)
    #load decoded audio file into librosa
    y, sr = librosa.load('a.wav', sr=22050)
    #delete audio file from system
    if os.path.exists("a.wav"):
        os.remove("a.wav")
    #establish preprocessing variables
    song_duration = int(librosa.core.samples_to_time(len(y), 22050))
    SAMPLE_RATE = 22050
    num_mfcc=13
    n_fft=2048
    hop_length=512
    #num_segments = int(song_duration/6)
    num_segments = int(len(y)/132300)
    SAMPLES_PER_TRACK = SAMPLE_RATE * song_duration
    #samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    samples_per_segment = 132300 #via trained keras model
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    #establish mfcc data dict for storage
    data = {
        'mfcc': []
        }
    genres = {
        'label': ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hiphop', 'country', 'jazz'],
        'id': [0,1,2,3,4,5,6,7,8,9]
        }
    #process mfcc audio data (extraction)
    for segment in range(num_segments):
        start = segment * samples_per_segment
        end = start + samples_per_segment
                
        mfcc = librosa.feature.mfcc(y[start:end], 
                                    sr, 
                                    n_mfcc=num_mfcc, 
                                    n_fft=n_fft, 
                                    hop_length=hop_length)
        mfcc = mfcc.T
                
        #                                                     * our data needs to be the same shape, therefor must contain a constant # of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
    #initialize data variables
    cP_max = 0
    p_genI = []

    #load model
    model = load_model('CNN_2.h5')

    #establish a max confidence and genre prediction accross entire song
    for s in data['mfcc']:
        X = np.array(s)
        X = X[..., np.newaxis]
        X = X[np.newaxis, ...]
        p = model.predict(X)
        p_index = np.argmax(p, axis=1)
        p_genre = genres['label'][p_index[0]]
        p_max = np.amax(p)
    
        cP_max = cP_max + p_max
        p_genI.append(p_index[0])

    cP_max = cP_max/len(data['mfcc'])
    m = stats.mode(p_genI)

    p_gen = genres['label'][m[0][0]]

    params = {
        'prediction' : {
            'genre': p_gen,
            'confidence': cP_max
        }
    }
    return jsonify(params)
    



        

