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
from model import EmbeddingModel,VoiceDataset,inference,GalleryDataset,ProbeDataset
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
import matplotlib.pyplot as plt

def computeGallery(galleryDataset,embeddingModel):
    d={}
    max=0
    for x in range(len(galleryDataset)):
        input, y = galleryDataset[x]
        input = input.unsqueeze(0) #Creates an added dimension of B = 1 (a batch with one element) so that the format is uniform
        input = input.to(device)

        a = inference(embeddingModel,input)
        if y in d:
            d[y].append(a)
        else: d[y]=[a]
        if y>max:max=y
    for key in d.keys():
        tot = None
        for elem in d[key]:
            if tot==None: tot = elem
            else: tot +=elem
        d[key] = tot/len(d[key])
    return (d,max)

def json_to_dict_tensors(json_path: str) -> dict[int, torch.Tensor]:
    """
    Convert JSON back to {int: torch.Tensor}.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    result = {}
    for k, v in raw.items():
        tensor = torch.tensor(v["data"], dtype=getattr(torch, v["dtype"].split(".")[-1]))
        result[int(k)] = tensor.reshape(v["shape"])

    return result

def dict_tensors_to_json(data: dict[int, torch.Tensor], json_path: str):
    """
    Convert {int: torch.Tensor} to JSON.
    """
    serializable = {
        str(k): {
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "data": v.tolist()
        }
        for k, v in data.items()
    }

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

def recognize(a,b):
    a=a.cpu()
    b=b.cpu()
    return (np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b.T)))

def analyze(probeSet,gallery,Threshold,embeddingModel,max):
    CorrectIdentifications=0
    ErroneousIdentifications=0
    CorrectRejections=0
    ErroneousRejections=0
    for x in range(len(probeSet)):
        input, expectedId = probeSet[x]
        input = input.unsqueeze(0) #Creates an added dimension of B = 1 (a batch with one element) so that the format is uniform
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input=input.to(device)
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
            if expectedId>max: CorrectRejections+=1
            else: ErroneousRejections+=1
    
    print(CorrectIdentifications,ErroneousIdentifications,CorrectRejections,ErroneousRejections)
    TP = CorrectIdentifications
    TN = CorrectRejections
    FP = ErroneousIdentifications
    FN = ErroneousRejections

    def safe_div(num, den):
        return num / den if den != 0 else 0.0

    TP_rate = safe_div(TP, TP + FN)
    TN_rate = safe_div(TN, TN + FP)
    FP_rate = safe_div(FP, FP + TN)
    FN_rate = safe_div(FN, FN + TP)
    ACCURACY = safe_div(TP + TN, TP + TN + FP + FN)

    # Biometric metrics
    DIR = safe_div(TP, TP + FN)         # Detection and Identification Rate
    FNIR = safe_div(FN, TP + FN)       # False Negative Identification Rate
    FPIR = safe_div(FP, FP + TN)       # False Positive Identification Rate

    # Print everything
    print(f"TP rate (Recall): {TP_rate:.4f}")
    print(f"TN rate (Specificity): {TN_rate:.4f}")
    print(f"FP rate: {FP_rate:.4f}")
    print(f"FN rate: {FN_rate:.4f}")
    print(f"Accuracy: {ACCURACY:.4f}")
    print(f"DIR: {DIR:.4f}")
    print(f"FPIR: {FPIR:.4f}")
    print(f"FNIR: {FNIR:.4f}")

    # Return all metrics
    return TP_rate, TN_rate, FP_rate, FN_rate, ACCURACY, DIR, FPIR, FNIR


def getModel():
    MEAN, STD =-50.573315, 17.590815
    TRAIN_SAVE_WEIGHTS = "C:\\Users\\Mattia\\Documents\\biometric\\train_weights\\best.pt"
    PROCESSED_DATASET_PATH = "C:\\Users\\Mattia\\Documents\\biometric\\train_npy\\train_npy\\npy"
    EMBEDDING_SIZE = 256

    trainDataset = VoiceDataset(PROCESSED_DATASET_PATH,MEAN,STD)
    embeddingModel = EmbeddingModel(EMBEDDING_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddingModel.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddingModel = EmbeddingModel(EMBEDDING_SIZE)
    embeddingModel.to(device)

    #loss = CosFaceLoss(EMBEDDING_SIZE,num_classes).to(device)

    weights = torch.load(TRAIN_SAVE_WEIGHTS,map_location=device)

    embeddingModel.load_state_dict(weights["model_state_dict"])
    return embeddingModel

def split(filepath):
    users = [id for id in listdir(filepath) if not isfile(join(filepath, id))]
    ret={}
    for id in users:
        ret[id] = ([],[])
        left=0
        right=0
        localPath = join(filepath, id)
        clips = [id for id in listdir(localPath) if not isfile(join(localPath, id))]
        d={}
        for clip in clips:
            clipsInSegment=0
            clipPath = join(localPath,clip)
            segments = [id for id in listdir(clipPath) if isfile(join(clipPath, id))]
            for segment in segments:
                clipsInSegment+=1
            if clipsInSegment in d.keys():
                d[clipsInSegment].append(clip)
            else: d[clipsInSegment]=[clip]
        od = OrderedDict(sorted(d.items(),reverse=True))
        for k,v in od.items():
            for clip in v:
                if left<=right:
                    left+=k
                    ret[id][0].append(clip)
                else:
                    right+=k
                    ret[id][1].append(clip)
                    
    return ret

def analyze_over_thresholds(probeDataset,gallery,embeddingModel,max,thresholds):
    """
    Runs analyze() for multiple thresholds.
    """
    results = {
        "threshold": [],
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": [],
        "Accuracy": [],
        "DIR": [],
        "FPIR": [],
        "FNIR": []
    }

    for t in thresholds:
        print(t)
        TP, TN, FP, FN, ACC, DIR, FPIR, FNIR = analyze(probeDataset,gallery,t,embeddingModel,max)
        results["threshold"].append(t)
        results["TP"].append(TP)
        results["TN"].append(TN)
        results["FP"].append(FP)
        results["FN"].append(FN)
        results["Accuracy"].append(ACC)
        results["DIR"].append(DIR)
        results["FPIR"].append(FPIR)
        results["FNIR"].append(FNIR)

    return results

def plot_results(results):
    plt.figure(figsize=(8, 5))
    plt.plot(results["threshold"], results["DIR"], label="DIR")
    plt.plot(results["threshold"], results["FPIR"], label="FPIR")
    plt.plot(results["threshold"], results["FNIR"], label="FNIR")

    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("Confusion Matrix Values vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

def plotMetrics(results):
    plt.figure(figsize=(8, 5))
    plt.plot(results["threshold"], results["TP"], label="TPR")
    plt.plot(results["threshold"], results["TN"], label="TNR")
    plt.plot(results["threshold"], results["FP"], label="FPR")
    plt.plot(results["threshold"], results["FN"], label="FNR")
    plt.plot(results["threshold"], results["Accuracy"], label="Accuracy", linewidth=2)


    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("Performance Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()


def plotROCRatio(results):
    """
    Plots ROC using DIR (Y-axis) vs FPIR (X-axis) as a ratio.
    Expects results dict with keys: 'TP', 'TN', 'FP', 'FN'.
    """
    DIR = []
    FPIR = []

    for TP, TN, FP, FN in zip(results["TP"], results["TN"], results["FP"], results["FN"]):
        # DIR = TP / (TP + FN)
        dir_val = TP / (TP + FN) if (TP + FN) > 0 else 0
        # FPIR = FP / (FP + TN)
        fpir_val = FP / (FP + TN) if (FP + TN) > 0 else 1e-6  # avoid division by zero

        DIR.append(dir_val)
        FPIR.append(fpir_val)

    plt.figure(figsize=(8, 6))
    plt.plot(FPIR, DIR, marker='o', linestyle='-', color='b', label="ROC (DIR vs FPIR)")
    plt.xlabel("FPIR (False Positive Identification Rate)")
    plt.ylabel("DIR (Detection and Identification Rate)")
    plt.title("Open Set Identification (DIR vs FPIR)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__=="__main__":

    TRAIN_SAVE_WEIGHTS = "C:\\Users\\Mattia\\Documents\\biometric\\train_weights\\best.pt"
    PROCESSED_GALLERY_PATH = "C:\\Users\\Mattia\\Documents\\biometric\\gallery_probeKnown_npy_validation\\gallery_probeKnown_npy_validation\\npy"
    PROCESSED_PROBE_PATH = "C:\\Users\\Mattia\\Documents\\biometric\\gallery_probeKnown_npy_validation\\gallery_probeKnown_npy_validation\\npy"

    embeddingModel=getModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splitData = split("./gallery_probeKnown_npy_validation/gallery_probeKnown_npy_validation/npy")

    galleryDataset = GalleryDataset(PROCESSED_GALLERY_PATH,splitData)

    gallery = json_to_dict_tensors("./gallery.json")
    if len(gallery.keys())==0:
        print("computing gallery")
        gallery = computeGallery(galleryDataset,embeddingModel)
        max = gallery[1]
        gallery=gallery[0]
        dict_tensors_to_json(gallery,"./gallery.json")
    else:
        max=0
        for k,v in gallery.items():
            if k>max: max=k
    print(max)
    probeDataset = ProbeDataset(PROCESSED_PROBE_PATH,PROCESSED_GALLERY_PATH,splitData)
    thresholds = np.arange(0, 1.0001, 0.025)
    res = analyze_over_thresholds(probeDataset,gallery,embeddingModel,max,thresholds)
    plot_results(res)
    plotMetrics(res)
    plotROCRatio(res)
