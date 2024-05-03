
"""
This file can be used to try a live prediction. 
"""

import keras
import numpy as np
import librosa

class livePredictions:
    """
    Main class of the application.
    """

    def __init__(self, path, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.path = path
        self.file = file

    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=0)
        predictions = self.loaded_model.predict(x)
        predicted_class = np.argmax(predictions, axis=1)  # Getting the index of the highest probability
        emotion_label = self.convertclasstoemotion(predicted_class)
        print("Prediction is:", emotion_label)


    @staticmethod
    def convertclasstoemotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

pred = livePredictions(path="D://SER//SmartMusicPlayer//Emotion_Voice_Detection_Model1.h5",file="D:SER//RAVDESS//Actor_08/03-02-06-02-02-01-08.wav")

pred.load_model()
pred.makepredictions()