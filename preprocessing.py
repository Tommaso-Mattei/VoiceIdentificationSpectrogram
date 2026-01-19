import torch
import torchvision.transforms as transforms
from pathlib import Path
import librosa
import cv2
import numpy as np
import os


def transform_wav_to_spectrogram(audioFilePath):
      data, samplerate = librosa.load(audioFilePath)

      trimmed_data, _ = librosa.effects.trim(data)

      time_duration = 10 # 10 seconds of audio (given the new sampling rate from librosa.load which is a default number)

      max_length = time_duration*samplerate

      if len(trimmed_data) > max_length:
        trimmed_data = trimmed_data[:max_length]
      else:
        trimmed_data = np.tile(trimmed_data,5) #minimum of audio length is approximately 2 seconds = 2*5 = 10
        trimmed_data = trimmed_data[:max_length]

      melSpectrogram = librosa.feature.melspectrogram(y=trimmed_data, sr= samplerate)

      decibel_image = librosa.power_to_db(melSpectrogram, ref=np.max)

      return decibel_image, samplerate


def transform_decibel_to_final(decibel_image,mean,std):
  normalized_image = (decibel_image - mean)/std

  resized_img = cv2.resize(normalized_image, (288, 432))

  x = np.stack([resized_img]*3, axis=-1) #the shape is (432,288) ---> (432,288,3) by having each element repeated thrice, just like a rgb image
  return x


def preprocessing_dataset(voice_dataset_path,processed_dataset_path,mean,std):
  voice_dataset_path = Path(voice_dataset_path)
  processed_dataset_path = Path(processed_dataset_path)

  for id_person in voice_dataset_path.iterdir():
    target_id_folder = processed_dataset_path / id_person.name #it's a join
    target_id_folder.mkdir(parents=True, exist_ok=True)
    for video in id_person.iterdir():
      target_video_folder = target_id_folder / video.name
      target_video_folder.mkdir(parents=True, exist_ok=True)
      for wav_file in video.iterdir():
        if wav_file.is_file():
          decibel_img, samplerate = transform_wav_to_spectrogram(wav_file)
          final_image = transform_decibel_to_final(decibel_img,mean,std)
          npy_path = target_video_folder / (wav_file.stem + ".npy") #.stem = only file name without extension
          np.save(npy_path, final_image.astype(np.float32))
  print("Finished")


def calculate_Std_Mean(soundList):
  all_spectrograms = []

  for element in soundList:
    all_spectrograms.append(transform_wav_to_spectrogram(element[0]))

  all_values = np.concatenate([spectrogram.flatten() for spectrogram in all_spectrograms])

  mean = all_values.mean()
  std  = all_values.std() + 1e-16

  return mean, std