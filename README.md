# EE800 for Jiahao Lu

This project is mainly for voiceprint recognition, which is also known as speaker recognition. This project includes the training of custom datasets, voiceprint comparison, and voiceprint recognition.

## Training

Before performing training, several parameters in `train.py` may need to be modified.
 - `gpu` is to specify which GPUs are used, and how to use all GPUs in case of multiple cards.
 - `resume` is used to resume training. If you have a previously trained model, you can use this parameter to specify the path of the model and resume training.
 - `batch_size` sets the size of the batch according to the size of your own video memory.
 - `n_classes` is the number of classifications, this can be viewed in the previous step to generate a list of data last to get the number of classifications, but also remember to add 1, because the label is from 0.
 - `multiprocess` this parameter is to specify how many threads to use to read the data, because reading audio needs to be slower, training is also trained by default using 4 multi-threads, so if you use multi-threads to read the data, do not use multi-threads to read the data, or vice versa, it is best to use multi-threads to read the data under Ubuntu. But Windows does not support multiple threads to read data, and it must be 0 under Windows.
 - The `net` parameter specifies the model to use, there are two models to choose from, the smaller resnet34s, and the larger resnet34l.
 
Finally, `train.py` is executed to start the training. During the training process, the model is saved at each step, and the logs information of the training is also recorded using Tensorboard.

# Predictions

This project provides two prediction schemes.

 - The first one is voiceprint comparison `predict_contrast.py`, which is to compare the voiceprint similarity of two audios, where the parameters `audio1_path` and `audio2_path` are the audio paths to be compared, and the other parameters need to be consistent with the training ones.
 - The second one is `predict_recognition.py`, which identifies the speaker belongs to the one in the voiceprint database by the recording, and outputs the name of the speaker and the recognition degree compared with the voiceprint database, and the other parameters need to be consistent with the training ones.
