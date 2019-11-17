import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import pickle as pkl
import os

def getSamplingRateFromFile(filename):
    song = AudioSegment.from_file(filename)
    return song.frame_rate


def calculateTempoFromFile(filename, hop_length=512):
    sr = getSamplingRateFromFile(filename)
    y, sr = librosa.core.load(filename,sr)
    tempo = librosa.beat.tempo(y=y, sr=sr, aggregate=None, hop_length=hop_length)
    return tempo

# def calculateDerivativeTempo(tempo):
#     dtempo = [tempo[i+1] - tempo[i] for i in range(len(tempo)-1)]
#     return np.array(dtempo)

##Reference : https://librosa.github.io/librosa_gallery/auto_examples/plot_vocal_separation.html
def getVocalPartsFromSignal(y,sr):
    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)


    margin_v = 2
    power = 2
    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)
    S_foreground = mask_v * S_full
    D_foreground = S_foreground * phase
    y_foreground = librosa.istft(D_foreground)
    return y_foreground

def plotFeatureTimeGraph(feature, hop_length, sr):

    x = [i*hop_length/sr for i in range(len(feature))]
    y = list(feature)
    plt.plot(x,y)
    plt.show()

def getNoiseRemovedVoiceParts(voice_parts, thresh, sr, hop_length=1024):
    frame_length = hop_length
    rms = librosa.feature.rms(voice_parts, frame_length=frame_length, hop_length=hop_length)
    rms = rms.T
    for i in range(rms.shape[0]):
        if(rms[i] < thresh):
            end = min((i+1)*hop_length, voice_parts.shape[0])
            voice_parts[i*hop_length : end] = 0.0
    
    librosa.output.write_wav('temp/noise_removed.wav', voice_parts, sr)
    return voice_parts

def SaveVoiceParts(filename,thresh,voiceSameTime=0.075,hop_length=1024):
    
    sr = getSamplingRateFromFile(filename)
    y,sr = librosa.core.load(filename,sr)
    print('number of samples ',len(y))
    voice_parts = getVocalPartsFromSignal(y,sr)
    fin_voice_parts = getNoiseRemovedVoiceParts(voice_parts, thresh, sr, hop_length=hop_length)
    print('number of voice samples',len(fin_voice_parts))
    fin_voice_parts[fin_voice_parts > 0] = 1
    voice_times_list = []
    startTime = 0
    endTime = 0
    voiceSameFrames = sr * voiceSameTime
    count = 0
    i = 0
    while(i < len(fin_voice_parts)):
        if(fin_voice_parts[i] == 1):
            startTime = float(i)/sr
            count = 0
            while(count < voiceSameFrames and i < len(fin_voice_parts)):
                if(fin_voice_parts[i] == 1):
                    count = 0
                elif(fin_voice_parts[i] == 0):
                    count += 1
                i+=1
            endTime = float(i)/sr
            voice_times_list.append((startTime,endTime))
        i+=1

    for i in voice_times_list:
        print(i)
    
    voice_parts_file = 'temp/' + os.path.basename(filename) + '_voice_parts.vp'
    ifile = open(voice_parts_file,'wb')
    pkl.dump(voice_times_list,ifile)
    ifile.close()
    

     


#working parameters are 
    #Energy threshold : 0.035
    #hop length = 512
    





if __name__ == '__main__':
    hop_length = 1024
    thresh = 0.035
    filename = 'Dataset/sunflower.mp4'
    SaveVoiceParts(filename, thresh, voiceSameTime=0.075, hop_length=hop_length)





