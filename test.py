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
import json
from model import EmbeddingModel,VoiceDataset,inference

def computeGallery(galleryDataset,embeddingModel):
    d={}
    for x in range(len(galleryDataset)):
        input, y = galleryDataset[x]
        input1 = input1.unsqueeze(0) #Creates an added dimension of B = 1 (a batch with one element) so that the format is uniform
        input1 = input1.to(device)

        a = inference(embeddingModel,input)
        if y in d:
            d[y].append(a)
        else: d[y]=[a]
    for key in d.keys():
        tot = None
        for elem in d[key]:
            if tot==None: tot = elem
            else: tot +=elem
        d[key] = tot/len(d[key])
    return d

def json_to_ndarray_dict(filepath: str) -> dict:
    """
    Load a JSON file and reconstruct a dictionary of NumPy ndarrays.
    """
    with open(filepath, "r") as f:
        serialized = json.load(f)

    data = {}
    for key, value in serialized.items():
        arr = np.array(value["data"], dtype=value["dtype"])
        arr = arr.reshape(value["shape"])
        data[key] = arr

    return data

def ndarray_dict_to_json(data: dict, filepath: str):
    """
    Save a dictionary of NumPy ndarrays to a JSON file.
    """
    serializable = {}

    for key, arr in data.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Value for key '{key}' is not a NumPy ndarray")

        serializable[key] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "data": arr.tolist()
        }

    with open(filepath, "w") as f:
        json.dump(serializable, f)


def recognize(a,b):
    return (np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b.T)))

def analyze(probeSet,gallery,Threshold):
    CorrectIdentifications=0
    ErroneousIdentifications=0
    CorrectRejections=0
    ErroneousRejections=0
    for x in range(len(probeSet)):
        input, expectedId = probeSet[x]
        newUser = inference(embeddingModel,input)
        bestMatch = (0,None)
        for id,val in gallery.items():
            similarity = recognize(newUser,val)
            if similarity>bestMatch[0]:
                bestMatch=(similarity,id)
        if bestMatch[0]>Threshold:
            if bestMatch[1]==expectedId: CorrectIdentifications+=1
            else:ErroneousIdentifications+=1
        else: 
            if expectedId==None: CorrectRejections+=1
            else: ErroneousRejections+=1
    total = CorrectIdentifications+CorrectRejections+ErroneousIdentifications+ErroneousRejections
    print(CorrectIdentifications,ErroneousIdentifications,CorrectRejections,ErroneousRejections)
    print("TP rate: "+str(CorrectIdentifications/total))
    print("TN rate: "+str(CorrectRejections/total))
    print("FP rate: "+str(ErroneousIdentifications/total))
    print("FN rate: "+str(ErroneousRejections/total))
    return (CorrectIdentifications,ErroneousIdentifications,CorrectRejections,ErroneousRejections)



if __name__=="__main__":

    MEAN, STD =-50.573315, 17.590815
    TRAIN_SAVE_WEIGHTS = "C:\\Users\\Mattia\\Documents\\biometric\\train_weights\\best.pt"
    PROCESSED_DATASET_PATH = "C:\\Users\\Mattia\\Documents\\biometric\\train_npy\\train_npy\\npy"
    BATCH_SIZE = 64
    EMBEDDING_SIZE = 256
    EPOCHS = 1
    LR = 0.0001

    trainDataset = VoiceDataset(PROCESSED_DATASET_PATH,MEAN,STD)
    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    num_classes = trainDataset.num_classes  
    embeddingModel = EmbeddingModel(EMBEDDING_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddingModel.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddingModel = EmbeddingModel(EMBEDDING_SIZE)
    embeddingModel.to(device)

    #loss = CosFaceLoss(EMBEDDING_SIZE,num_classes).to(device)

    weights = torch.load(TRAIN_SAVE_WEIGHTS,map_location=device)

    embeddingModel.load_state_dict(weights["model_state_dict"])


    galleryDataset = trainDataset

    input, y = trainDataset[0]
    input = input.unsqueeze(0) #Creates an added dimension of B = 1 (a batch with one element) so that the format is uniform
    input = input.to(device)

    embedding = inference(embeddingModel,input).cpu()

    
    gallery = json_to_ndarray_dict("./gallery.json")
    if len(gallery.keys())==0:
        gallery = computeGallery(galleryDataset,embeddingModel)
        ndarray_dict_to_json(gallery,"./gallery.json")
    
    probeDataset = trainDataset[:1000]
    analyze(probeDataset,gallery,0.5)
    
