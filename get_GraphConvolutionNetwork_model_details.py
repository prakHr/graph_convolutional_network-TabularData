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
    adjacency_list = []
    for smallX in X:
        my_dict = {
            "X":X,
            "smallX":smallX,
            "threshold":threshold
        }
        zero_filled_list = [0 for i in range(X.shape[0])]
        adjacency_list.append(zero_filled_list)
        results.append(my_dict)
    num_cores = max(multiprocessing.cpu_count()//2,1)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(get_similar_values, results, progress_bar=False)
    for i in range(len(results)):
        from_node = i
        to_nodes_list = list(list(results[i])[0])
        for to_node in to_nodes_list:        
            adjacency_list[from_node][to_node] = 1
            adjacency_list[to_node][from_node] = 1
            adjacency_list[to_node][to_node] = 0
        adjacency_list[from_node][from_node] = 0
    return adjacency_list

    
def get_graph_details(X,y,epochs,threshold):
    num_samples,num_features = X.shape[0],X.shape[1]
    n_classes = len(list(set(list(y))))
    adjacency_list = construct_adjacency_list(X,threshold)
    adjacency_list = np.array(adjacency_list)
    hidden_layers_lambda_dimensions = 5
    graph_convolutional_network = GCN(in_features = num_features, hidden = hidden_layers_lambda_dimensions*num_features, n_classes=n_classes)
    graph_convolutional_network.fit(adjacency_list,X,y,epochs=epochs)
    return {
        "graph_convolutional_network":graph_convolutional_network,
        "threshold":threshold,
        "epochs":epochs,
        "A":adjacency_list
    }    


if __name__=="__main__":
    X = np.random.randn(4,8)
    y = np.array([0,0,1,1])
    epochs = 100000
    threshold = 0.001
    results = get_graph_details(X,y,epochs,threshold)
    pprint(results)
