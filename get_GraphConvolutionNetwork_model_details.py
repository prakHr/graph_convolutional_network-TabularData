from mandala_gnn import GCN
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mpire import WorkerPool
from pprint import pprint
import mpire
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import multiprocessing 
from mpire import WorkerPool
from pprint import pprint
from multiprocessing import Manager
import numpy as np

def get_similar_values(X,smallX,threshold):
    similarities = cosine_similarity(X, smallX.reshape(1, -1)).ravel()
    valid_indices = np.where(similarities >= threshold)
    return valid_indices

def construct_adjacency_list(X,threshold):
    results = []
    A = []
    for smallX in X:
        my_dict = {
            "X":X,
            "smallX":smallX,
            "threshold":threshold
        }
        l = [0 for i in range(X.shape[0])]
        A.append(l)
        results.append(my_dict)
    num_cores = max(multiprocessing.cpu_count()//2,1)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(get_similar_values, results, progress_bar=False)
    for i in range(len(results)):
        u = i
        vD = list(list(results[i])[0])
        for v in vD:        
            A[u][v] = 1
            A[v][u] = 1
            A[v][v] = 0
            A[u][u] = 0
    return A

    
def get_graph_details(X,y,epochs,threshold):
    N,D = X.shape[0],X.shape[1]
    x = len(list(set(list(y))))
    A = construct_adjacency_list(X,threshold)
    A = np.array(A)
    gcn = GCN(in_features = D, hidden = 2*D, n_classes=x)
    gcn.fit(A,X,y,epochs=epochs)
    return {
        "graph_convolutional_network":gcn,
        "threshold":threshold,
        "epochs":epochs,
        "A":A
    }    


if __name__=="__main__":
    X = np.random.randn(4,8)
    y = np.array([0,0,1,1])
    epochs = 2
    threshold = 0.91
    results = get_graph_details(X,y,epochs,threshold)
    pprint(results)
