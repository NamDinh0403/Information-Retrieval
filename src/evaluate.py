import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np

def calculate_map(qB, rB, query_L, retrieval_L):
    """
    Calculate Mean Average Precision (mAP)
    qB: Query Binary codes [-1, 1]
    rB: Retrieval Database Binary codes [-1, 1]
    """
    num_query = query_L.shape[0]
    map_sum = 0
    
    # Calculate hamming distance
    # 0.5 * (K - qB @ rB.T) is the hamming distance when vectors are -1, 1
    dist = 0.5 * (qB.shape[1] - np.dot(qB, rB.T))
    
    for i in range(num_query):
        gnd = (query_L[i] == retrieval_L).astype(np.float32)
        if np.sum(gnd) == 0:
            continue
        
        ord_indices = np.argsort(dist[i])
        gnd = gnd[ord_indices]
        
        tsum = np.sum(gnd)
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        
        map_sum += np.mean(count / (tindex))
        
    return map_sum / num_query

def evaluate(model, query_loader, db_loader, device):
    model.eval()
    qB_list = []
    qL_list = []
    
    rB_list = []
    rL_list = []
    
    print("Extracting query codes...")
    with torch.no_grad():
        for imgs, labels in tqdm(query_loader):
            imgs = imgs.to(device)
            # Use sign() function for binary codes during inference
            hash_codes, _ = model(imgs)
            binary_codes = torch.sign(hash_codes).cpu().numpy()
            qB_list.append(binary_codes)
            qL_list.append(labels.numpy())
            
    print("Extracting database codes...")
    with torch.no_grad():
        for imgs, labels in tqdm(db_loader):
            imgs = imgs.to(device)
            hash_codes, _ = model(imgs)
            binary_codes = torch.sign(hash_codes).cpu().numpy()
            rB_list.append(binary_codes)
            rL_list.append(labels.numpy())
            
    qB = np.concatenate(qB_list)
    qL = np.concatenate(qL_list)
    rB = np.concatenate(rB_list)
    rL = np.concatenate(rL_list)
    
    map_score = calculate_map(qB, rB, qL, rL)
    print(f"Evaluation mAP: {map_score:.4f}")
    return map_score
