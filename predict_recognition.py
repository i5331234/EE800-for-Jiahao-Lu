import argparse
import os
import time
import wave
import numpy as np
import pyaudio
import utils
import model
import tensorflow

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--n_classes', default=5994, type=int, help='class dim number')
parser.add_argument('--audio_db', default='audio_db/', type=str, help='person audio database')
parser.add_argument('--resume', default=r'pretrained/weights.h5', type=str, help='resume model path')
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
args = parser.parse_args()

person_feature = []
person_name = []

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
_ = tensorflow.Session(config=config)
# ==================================
#       Get Model
# ==================================
# construct the data generator.
params = {'dim': (257, None, 1),
          'nfft': 512,
          'spec_len': 250,
          'win_length': 400,
          'hop_length': 160,
          'n_classes': args.n_classes,
          'sampling_rate': 16000,
          'normalize': True}

network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                            num_class=params['n_classes'],
                                            mode='eval', args=args)

# ==> load pre-trained model
network_eval.load_weights(os.path.join(args.resume), by_name=True)
print('==> successfully loading model {}.'.format(args.resume))


def predict(audio_path):
    specs = utils.load_data(audio_path, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    feature = network_eval.predict(specs)[0]
    return feature


def load_audio_db(audio_db_path):
    start = time.time()
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = predict(path)
        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)
    end = time.time()
    print('Loading of audio library completed, time consuming：%fms' % (round((end - start) * 1000)))


def recognition(path):
    name = ''
    pro = 0
    feature = predict(path)
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f.T)
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro


def start_recognition():

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "infer_audio.wav"

    while True:
        # open recording
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        i = input("Press enter key to start recording, record %s in seconds：" % RECORD_SECONDS)
        print("start recording......")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("recording complete!")

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Identify and compare audio from audio libraries
        start = time.time()
        name, p = recognition(WAVE_OUTPUT_FILENAME)
        end = time.time()
        if p > 0.8:
            print("Prediction time: %d, recognition of speech: %s, similarity: %f" % (round((end - start) * 1000), name, p))
        else:
            print("The prediction time is: %d, the audio library does not have this user's voice" % round((end - start) * 1000))


if __name__ == '__main__':
    load_audio_db(args.audio_db)
    start_recognition()
