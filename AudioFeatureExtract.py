import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

def getSamplingRateFromFile(filename):
    song = AudioSegment.from_file(filename)
    return song.frame_rate

def getTotalSamples(filename,sr):
    y, sr = librosa.core.load(filename,sr=sr)
    return len(y)


def calculateTempoFromFile(filename, hop_length=512):
    sr = getSamplingRateFromFile(filename)
    y, sr = librosa.core.load(filename,sr)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    print('onset samples ', len(onset_env))
    tempo = librosa.beat.tempo(y=y, sr=sr, aggregate=None, hop_length=hop_length)
    print('total samples ',len(y))
    print('tempo samples ',len(tempo))
    return tempo

def calculateDerivativeTempo(tempo):
    dtempo = [tempo[i+1] - tempo[i] for i in range(len(tempo)-1)]
    return np.array(dtempo)

##Reference : https://librosa.github.io/librosa_gallery/auto_examples/plot_vocal_separation.html
def getVocalPartsFromFile(filename,sr):
    y, sr = librosa.core.load(filename, duration=None,sr=sr)
    print('sampling rate is ',sr)
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
    librosa.output.write_wav('foreground.wav', y_foreground, sr)
    return y_foreground

def plotTempoTimeGraph(tempo, hop_length, sr, totSamples):

    x = [i*hop_length/sr for i in range(len(tempo))]
    y = list(tempo)
    plt.plot(x,y)
    plt.show()

def plotAudioTimeGraph(y, sr):
    y1 = []
    for i in y:
        if(abs(i) > 0.65):
            y1.append(i)
        else:
            y1.append(0)
    plt.plot([i/float(sr) for i in range(len(y1))], y1)
    plt.show()
    plt.plot([i/float(sr) for i in range(len(y))], y)
    plt.show()







if __name__ == '__main__':
    hop_length = 1024
    filename = 'see_you_again.mp4'
    sr = getSamplingRateFromFile(filename)
    # print('sampling rate is ',sr)
    # totSamples = getTotalSamples(filename,sr)
    # tempo = calculateTempoFromFile(filename,hop_length)
    # plotTempoTimeGraph(tempo, hop_length, sr,totSamples)
    voice_mag = getVocalPartsFromFile(filename,sr)
    #print('total samples ',totSamples)
    print('voice samples ',len(voice_mag))
    plotAudioTimeGraph(voice_mag, sr)




