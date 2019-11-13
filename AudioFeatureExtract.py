import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

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

def getNoiseRemovedVoiceParts(voice_parts, thresh,sr):
    frame_length = 512
    hop_length = 512
    rms = librosa.feature.rms(voice_parts, frame_length=frame_length, hop_length=hop_length)
    rms = rms.T
    for i in range(rms.shape[0]):
        if(rms[i] < thresh):
            end = min((i+1)*hop_length, voice_parts.shape[0])
            voice_parts[i*hop_length : end] = 0.0
    
    #librosa.output.write_wav('noise_removed.wav', voice_parts, sr)
    return voice_parts

def extractVoiceParts(filename,thresh):
    sr = getSamplingRateFromFile(filename)
    y,sr = librosa.core.load(filename,sr)
    voice_parts = getVocalPartsFromSignal(y,sr)
    fin_voice_parts = getNoiseRemovedVoiceParts(voice_parts, thresh, sr)
    fin_voice_parts[fin_voice_parts > 0] = 1
     

    





if __name__ == '__main__':
    hop_length = 512
    filename = 'temp/foreground.wav'
    sr = getSamplingRateFromFile(filename)
    # print('sampling rate is ',sr)
    # totSamples = getTotalSamples(filename,sr)
    # tempo = calculateTempoFromFile(filename,hop_length)
    # plotFeatureTimeGraph(tempo, hop_length, sr,totSamples)
    # voice_mag = getVocalPartsFromFile(filename,sr)
    # #print('total samples ',totSamples)
    # print('voice samples ',len(voice_mag))
    # plotAudioTimeGraph(voice_mag, sr)
    # voice_parts = getVocalPartsFromFile(filename,sr)
    voice_parts,sr = librosa.core.load(filename,sr)
    print(voice_parts.shape)
    rmse = librosa.feature.rms(voice_parts, frame_length=512, hop_length=512)
    print(rmse.shape)
    rmse = rmse.T
    plotFeatureTimeGraph(rmse, 512, sr)
    getNoiseRemovedVoiceParts(voice_parts, 0.025, sr)





