import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import Optimizer
torch.set_default_dtype(torch.float64)

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import json
import os
import cv2
import numpy as np
import datetime
import random
from typing import List, Dict, Tuple, Any, Union, Callable
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import itertools

import sys
sys.path.insert(1, "D:/Workspace/DATN/code")
from models.TPSGridGen.TPSGridGen import TPSGridGen
from losses import ProxyNCALoss, SoftTripleLoss, ProxyAnchorLoss, TripletMarginLoss
import pymongo
import matplotlib.pyplot as plt

import onnx, onnxruntime

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_current_time() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_config(config_root: str=None):

    with open(config_root, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config

def log_info_metrics(logger, metrics: Dict[str, float], current_epoch: int, current_iter: int) -> None:
    """
    Print all metrics to stdout
    """
    logger.info("*" * 130)
    logger.info(
        f"VALIDATING\t[{current_epoch}|{current_iter}]\t"
        f"MAP: {metrics['mean_average_precision']:.2f}%\t"
        f"AP@1: {metrics['average_precision_at_1']:.2f}%\t"
        f"AP@5: {metrics['average_precision_at_5']:.2f}%\t"
        f"Top-1: {metrics['top_1_accuracy']:.2f}%\t"
        f"Top-5: {metrics['top_5_accuracy']:.2f}%\t"
        f"NMI: {metrics['normalized_mutual_information']:.2f}\t"
    )
    logger.info("*" * 130)

def set_random_seed(seed: int) -> None:
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model: nn.Module,
                    config: Dict[str, Any],
                    current_epoch: int,
                    output_dir: str,
                    mean_average_precision: float = None,
                    ) -> str:

    checkpoint_name: str = f"epoch{current_epoch}"
    if mean_average_precision is not None:
        checkpoint_name += f"-map{mean_average_precision:.2f}"
    checkpoint_name += ".pth"

    checkpoint_path: str = os.path.join(output_dir, checkpoint_name)
    torch.save(
        {
            "model_state_dict": model.module.state_dict(),
        },
        checkpoint_path
    )
    return checkpoint_path

def calculate_all_metrics(model: nn.Module,
                          test_loader: DataLoader,
                          ref_loader: DataLoader,
                          device: torch.device,
                          k: Tuple[int, int, int] = (1, 5, 10)
                          ):

    # Calculate all embeddings of training set and test set
    embeddings_test, labels_test = get_embeddings_from_dataloader(test_loader, model, device)
    embeddings_ref, labels_ref = get_embeddings_from_dataloader(ref_loader, model, device)

    # Expand dimension for batch calculating
    embeddings_test = embeddings_test.unsqueeze(dim=0)  # [M x K] -> [1 x M x embedding_size]
    embeddings_ref = embeddings_ref.unsqueeze(dim=0)  # [N x K] -> [1 x N x embedding_size]
    labels_test = labels_test.unsqueeze(dim=1)  # [M] -> [M x 1]

    # Pairwise distance of all embeddings between test set and reference set
    distances: torch.Tensor = torch.cdist(embeddings_test, embeddings_ref, p=2).squeeze()  # [M x N]

    # Calculate precision_at_k on test set with k=1, k=5 and k=10
    metrics: Dict[str, float] = {}
    for i in k:
        metrics[f"average_precision_at_{i}"] = calculate_precision_at_k(distances,
                                                                        labels_test,
                                                                        labels_ref,
                                                                        k=i
                                                                        )
    # Calculate mean average precision (MAP)
    mean_average_precision: float = sum(precision_at_k for precision_at_k in metrics.values()) / len(metrics)
    metrics["mean_average_precision"] = mean_average_precision

    # Calculate top-1 and top-5 and top-10 accuracy
    for i in k:
        metrics[f"top_{i}_accuracy"] = calculate_topk_accuracy(distances,
                                                               labels_test,
                                                               labels_ref,
                                                               top_k=i
                                                               )
    # Calculate NMI score
    n_classes: int = len(test_loader.dataset.classes)
    metrics["normalized_mutual_information"] = calculate_normalized_mutual_information(
        embeddings_test.squeeze(), labels_test.squeeze(), n_classes
    )

    return metrics


def calculate_precision_at_k(distances: torch.Tensor,
                             labels_test: torch.Tensor,
                             labels_ref: torch.Tensor,
                             k: int
                             ) -> float:

    _, indices = distances.topk(k=k, dim=1, largest=False)  # indices shape: [M x k]

    y_pred = []
    for i in range(k):
        indices_at_k: torch.Tensor = indices[:, i]  # [M]
        y_pred_at_k: torch.Tensor = labels_ref[indices_at_k].unsqueeze(dim=1)  # [M x 1]
        y_pred.append(y_pred_at_k)

    y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
    labels_test = torch.hstack((labels_test,) * k)  # [M x k]

    precision_at_k: float = ((y_pred == labels_test).sum(dim=1) / k).mean().item() * 100
    return precision_at_k


def calculate_topk_accuracy(distances: torch.Tensor,
                            labels_test: torch.Tensor,
                            labels_ref: torch.Tensor,
                            top_k: int
                            ) -> float:

    _, indices = distances.topk(k=top_k, dim=1, largest=False)  # indices shape: [M x k]

    y_pred = []
    for i in range(top_k):
        indices_at_k: torch.Tensor = indices[:, i]  # [M]
        y_pred_at_k: torch.Tensor = labels_ref[indices_at_k].unsqueeze(dim=1)  # [M x 1]
        y_pred.append(y_pred_at_k)

    y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
    labels_test = torch.hstack((labels_test,) * top_k)  # [M x k]

    n_predictions: int = y_pred.shape[0]
    n_true_predictions: int = ((y_pred == labels_test).sum(dim=1) > 0).sum().item()
    topk_accuracy: float = n_true_predictions / n_predictions * 100
    return topk_accuracy


def calculate_normalized_mutual_information(embeddings: torch.Tensor,
                                            labels_test: torch.Tensor,
                                            n_classes: int
                                            ) -> float:
    embeddings = embeddings.cpu().numpy()
    y_test: np.ndarray = labels_test.cpu().numpy().astype(np.int)

    y_pred: np.ndarray = KMeans(n_clusters=n_classes).fit(embeddings).labels_
    NMI_score: float = normalized_mutual_info_score(y_test, y_pred)

    return NMI_score

def get_embedding(model: torch.nn.Module,
                  image_path: str,
                  transform: Callable,
                  device: torch.device
                  ) -> np.ndarray:
    image: Image.Image = Image.open(image_path).convert("RGB")
    input_tensor: torch.Tensor = transform(image).unsqueeze(dim=0).to(device)
    embedding: torch.Tensor = model(input_tensor)
    return embedding.detach().cpu().numpy()

@torch.no_grad()
def get_embeddings_from_dataloader(loader: DataLoader,
                                   model: nn.Module,
                                   device: torch.device,
                                   return_numpy_array=False,
                                   return_image_paths=False,
                                   ):
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[torch.Tensor] = []
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        labels: torch.Tensor = labels_.to(device, non_blocking=True)
        embeddings: torch.Tensor = model(images)
        embeddings_ls.append(embeddings)
        labels_ls.append(labels)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]
    labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]

    if return_numpy_array:
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()

    if return_image_paths:
        images_paths: List[str] = []
        for path, _ in loader.dataset.samples:
            images_paths.append(path)
        return (embeddings, labels, images_paths)

    return (embeddings, labels)

def train_one_batch(model: nn.Module,
                    optimizer: Optimizer,
                    loss_function: Union[TripletMarginLoss, ProxyNCALoss, ProxyAnchorLoss, SoftTripleLoss],
                    images: torch.Tensor,
                    labels: torch.Tensor,
                    device: torch.device,
                    ) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()

    images: torch.Tensor = images.to(device, non_blocking=True)
    labels: torch.Tensor = labels.to(device, non_blocking=True)

    embeddings: torch.Tensor = model(images)
    loss, fraction_hard_triplets = loss_function(embeddings, labels)

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "fraction_hard_triplets": float(fraction_hard_triplets)
    }

def loadROIModel(weight_path: str=None):
    assert os.path.exists(weight_path), "weight_path {} does not exists !".format(weight_path)
    model = ROILAnet()
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    model.requires_grads=False
    return model

def getThinPlateSpline(target_width: int = 112, target_height: int = 112) -> torch.Tensor:
    target_control_points = torch.Tensor(list(itertools.product(
        torch.arange(-1.0, 1.00001, 1.0),
        torch.arange(-1.0, 1.00001, 1.0),
    )))
    gridgen = TPSGridGen(target_height=target_height, target_width=target_width, target_control_points=target_control_points)
    gridgen = gridgen.to(device)
    return gridgen

def getOriginalAndResizedInput(path: str):
    if path is None:
        return (None, None)
    
    #define transformer for resized input of feature extraction CNN
    resizeTranformer = transforms.Compose([
            transforms.Resize((56,56)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    PILMain = Image.open(path).convert("RGB") # load image in PIL format
    
    sourceImage = np.array(PILMain).astype("float64") # convert from PIL to float64
    sourceImage = transforms.ToTensor()(sourceImage).unsqueeze_(0) # add first dimension, which is batch dim
    sourceImage = sourceImage.to(device) # load to available device

    resizedImage = resizeTranformer(PILMain)
    resizedImage = resizedImage.view(-1,resizedImage.size(0),resizedImage.size(1),resizedImage.size(2))
    resizedImage = resizedImage.to(device) # load to available device
    return (PILMain, sourceImage,resizedImage)

def getThetaHat(resizedImage: torch.Tensor = None, model = None): 
    if resizedImage is None:
        return None
        
    input = resizedImage.cpu().detach().numpy()
    data = json.dumps({"data": input.tolist()})
    data = np.array(json.loads(data)["data"]).astype("float32")
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    theta_hat = session.run([output_name], {input_name: data})
    theta_hat = theta_hat[0]
    theta_hat = torch.from_numpy(theta_hat)
    theta_hat = theta_hat.view(-1, 2, 9) # split into x and y vector -> theta_hat is originally a vector like [xxxxxxxxxyyyyyyyyy]
    theta_hat = torch.stack((theta_hat[:,0], theta_hat[:,1]),-1)
    return theta_hat

def sampleGrid(theta_hat: torch.Tensor = None, sourceImage: torch.Tensor = None, target_width: int = 112, target_height: int = 112 ) -> torch.Tensor:
    gridgen = getThinPlateSpline(target_width, target_height)
    #generate grid from calculated theta_hat vector
    source_coordinate = gridgen(theta_hat)
    #create target grid - with target height and target width
    grid = source_coordinate.view(-1, target_height, target_width, 2).to(device)
    #sample ROI from input image and created T(theta_hat)
    target_image = F.grid_sample(sourceImage, grid, align_corners=False)
    return target_image

def printExtraction(target_image: torch.Tensor = None, source_image = None):
    target_image = target_image.cpu().data.numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2)
    target_image = Image.fromarray(target_image.astype('uint8'))
    plt.imshow(source_image)
    plt.show() # show original image
    plt.imshow(target_image)
    plt.show() # show ROI

def getIROI(model, input):
    resizedImage = F.interpolate(input, (56, 56))
    theta_hat = getThetaHat(resizedImage=resizedImage, model=model) # create theta hat with normlized ROI coordinates
    IROI = sampleGrid(theta_hat=theta_hat, sourceImage=input, target_width=224, target_height=224) # get ROI from source image
    IROI.to(device)
    return IROI

def markImage(image, theta):
    nimg = np.array(image)
    ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx, coord in enumerate(theta):
        currX = coord[0]
        currY = coord[1]
        x = int((ocvim.shape[1] - 1) / (1 + 1) * (currX - 1) + ocvim.shape[1])
        y = int((ocvim.shape[0] - 1) / (1 + 1) * (currY - 1) + ocvim.shape[0])
        ocvim = cv2.circle(ocvim,(x,y),6,(200,0,0),2)
        ocvim = cv2.putText(
            ocvim,  # numpy array on which text is written
            str(idx),  # text
            (x, y),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            1,  # font size
            (209, 80, 0, 255),  # font color
            3)
    ocvim = ocvim[...,::-1]
    return ocvim
# @torch.no_grad()
def embedding_extraction(image_path: str, device: torch.device, transforms: Callable, draw: bool, name: str):
    import sys
    sys.path.insert(1, "D:/Workspace/Graduation_Project/sourceCode/DATN")
    if draw:
        fig = plt.figure(figsize=(15,15)) # specifying the overall grid size
    # model.eval()
    inputPIL = Image.open(image_path).convert('RGB')
    if draw:
        plt.subplot(1,3,1)
        plt.imshow(inputPIL)
        plt.title('Name: {}'.format(name))
    (PILMain, sourceImage,resizedImage) = getOriginalAndResizedInput(image_path)
    sourceImage = torch.stack([sourceImage.squeeze()])
    resizedImage = torch.stack([resizedImage.squeeze()])
    theta_hat = getThetaHat(resizedImage, "../models/onnx/ROI.onnx")
    if draw:
        plt.subplot(1,3,2)
        plt.imshow(markImage(inputPIL, theta_hat[0]))
        plt.title('Hand Landmarks')
    IROI = sampleGrid(theta_hat=theta_hat, sourceImage=sourceImage, target_width=300, target_height=300)
    IROI = IROI[0]
    if draw:
        plt.subplot(1,3,3)
        plt.imshow((IROI.cpu()[0]),cmap='gray')
        plt.title('Extracted ROI')
    roi = Image.fromarray(np.uint8(IROI.cpu()[0])).convert("RGB")
    input_tensor = transforms(roi).unsqueeze(dim=0).to(device)
    input = input_tensor.cpu().detach().numpy()
    data = json.dumps({"data": input.tolist()})
    data = np.array(json.loads(data)["data"]).astype("float32")
    session = onnxruntime.InferenceSession("../models/onnx/mobilenet.onnx", None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    if draw:
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        plt.plot()
    if draw:
        return session.run([output_name], {input_name: data}), fig
    return session.run([output_name], {input_name: data})

def get_collection(collection_name: str):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["myDatabase"]
    return mydb[collection_name]
