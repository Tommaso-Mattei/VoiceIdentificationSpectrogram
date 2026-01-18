import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
import librosa
import cv2
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

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


class CosFaceLoss(nn.Module):

    def __init__(self, embedding_size, num_classes, s = 22.0, m = 0.2):

        super(CosFaceLoss, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size)) #NxE
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels): #embeddings.shape = BxE
        normalized_embeddings = torch.nn.functional.normalize(embeddings)
        normalized_weights = torch.nn.functional.normalize(self.weight)


        #BxE * ExN = BxN
        cosine_similarity = torch.matmul(normalized_embeddings, normalized_weights.T) #dot product of L2 normalized vectors is the cosine similarity

        one_hot = torch.zeros_like(cosine_similarity)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        cosine_similarity_final = cosine_similarity - (one_hot*self.m)

        logits = cosine_similarity_final * self.s

        loss = nn.functional.cross_entropy(logits,labels)

        return loss

class VoiceDataset(Dataset):

  def __init__(self, voice_dataset_path, mean, std): #path of the train part of the dataset
    super().__init__()

    self.voice_dataset_path = Path(voice_dataset_path)

    soundList = [] #list of tuples with path to the .wav and the corresponding label

    self.mean = mean
    self.std = std

    for id_person in self.voice_dataset_path.iterdir():
      label = id_person.name
      for video in id_person.iterdir():
        for voice_chunk in video.iterdir():
          if voice_chunk.is_file():
            soundList.append((voice_chunk,label))

    self.soundList = soundList

    ids = sorted({labels for _,labels in soundList})

    self.num_classes = len(ids)

    self.idMap = {id_str:idx for idx, id_str in enumerate(ids)}

    self.samplerate = 0

  def __len__(self):
    return len(self.soundList)

  def __getitem__(self, idx):
      npyFilePath = self.soundList[idx][0]   # path to .npy

      final_image = np.load(npyFilePath, mmap_mode="r")

      x = torch.from_numpy(final_image).permute(2, 0, 1) #the way pytorch uses dimensions

      stringLabel = self.soundList[idx][1]  # label associated to .wav

      y = self.idMap[stringLabel]

      return [x, y]


def inference(model,input):
    model.eval()
    with torch.no_grad():
        embedding = model(input)
    return embedding

def train_model(model,trainLoader,epochs,optimizer,scheduler,loss,device,weights_path):

    # keeps track of the losses
    losses = []
    best_loss = float("inf") #we want to find lower bound

    # Training the model
    for epoch in range(epochs):
        ######### TRAINING ##########
        model.train()
        loss.train()
        running_loss = 0  # To track loss for this epoch

        # Using tqdm for the progress bar
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader), leave=True)

        for batch_idx, (data, targets) in loop:
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward pass
            embeddings = model(data)
            entropyLoss = loss(embeddings, targets)

            # Backward pass
            optimizer.zero_grad()
            entropyLoss.backward()

            # Gradient descent step
            optimizer.step()

            # Accumulate loss
            running_loss += entropyLoss.item()

            # Update progress bar with loss and epoch information
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=entropyLoss.item())

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(trainLoader)
        losses.append(avg_loss)

        #scheduler
        scheduler.step()

        # Print loss for this epoch
        tqdm.write(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "loss_state_dict": loss.state_dict()}, weights_path)

    print("It's finally over...")


class EmbeddingModel(nn.Module):

  def __init__(self, embedding_size):
    super().__init__()

    self.embedding_size = embedding_size

    self.model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # X dimensional vector from training gets reduced with a linear operation to our embedding size
    # normalizes within the batch
    self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, embedding_size), nn.BatchNorm1d(embedding_size))

  def forward(self, x):
    embeddings = self.model(x)

    normalized_embeddings = torch.nn.functional.normalize(embeddings)

    return normalized_embeddings

if __name__=="__main__":
    MEAN, STD =-50.573315, 17.590815
    TRAIN_SAVE_WEIGHTS = "C:\\Users\\Mattia\\Documents\\biometric\\train_weights\\best.pt"
    PROCESSED_DATASET_PATH = "C:\\Users\\Mattia\\Documents\\biometric\\train_npy\\train_npy\\npy"
    BATCH_SIZE = 32
    EMBEDDING_SIZE = 256
    EPOCHS = 10
    LR = 0.001

    trainDataset = VoiceDataset(PROCESSED_DATASET_PATH,MEAN,STD)
    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    num_classes = trainDataset.num_classes  
    embeddingModel = EmbeddingModel(EMBEDDING_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddingModel.to(device)

    loss = CosFaceLoss(EMBEDDING_SIZE,num_classes).to(device)

    optimizer = torch.optim.Adam(list(embeddingModel.parameters())+list(loss.parameters()), lr=LR)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    #Train ResNet

    print("train starting")
    train_model(embeddingModel,trainLoader, EPOCHS, optimizer, scheduler,loss,device,TRAIN_SAVE_WEIGHTS)

    embeddingModel = EmbeddingModel(EMBEDDING_SIZE)
    embeddingModel.to(device)

    #loss = CosFaceLoss(EMBEDDING_SIZE,num_classes).to(device)

    weights = torch.load(TRAIN_SAVE_WEIGHTS,map_location=device)

    embeddingModel.load_state_dict(weights["model_state_dict"])

    #loss.load_state_dict(weights["loss_state_dict"])
    input, y = trainDataset[0]
    input = input.unsqueeze(0) #Creates an added dimension of B = 1 (a batch with one element) so that the format is uniform
    input = input.to(device)

    embedding = inference(embeddingModel,input)
    print(embedding.shape)
