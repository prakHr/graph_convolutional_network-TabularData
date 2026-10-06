from transformers.utils import logging
from huggingface_hub import utils
# Disable model-loading progress bars
logging.disable_progress_bar()
utils.disable_progress_bars()
from mandala_gnn import MandalaGraph
import pandas as pd
import numpy as np
from pprint import pprint
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
from mandala_gnn import MandalaGraph
from bertopic import BERTopic
from mandala_gnn import CentralityAnalyzer, CommunityDetector, GraphMetrics

def create_scalable_edges_and_nodes(list_of_topic):
    edges = []
    nodes = []
    for i in range(len(list_of_topic)):
        for j in range(i+1, len(list_of_topic)):
            topic1 = list_of_topic[i]
            topic2 = list_of_topic[j]
            nodes.append(topic1)
            nodes.append(topic2)
            edges.append((topic1, topic2))
    return {
        "edges": edges,
        "nodes": nodes
    }

def create_knowledge_graph(list_of_sentences,topic_name_column):
    topic_model = BERTopic(verbose = False)
    docs = list_of_sentences*10
    topics, probs = topic_model.fit_transform(docs)
    topic_df = topic_model.get_topic_info()
    list_of_topics = topic_df[topic_name_column].to_list()
    results = []
    for list_of_topic in list_of_topics:
        my_dict = {
            "list_of_topic": list_of_topic,
        }
        results.append(my_dict)
    num_cores = max(multiprocessing.cpu_count()//2,1)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(create_scalable_edges_and_nodes, results, progress_bar=False)
    graph = MandalaGraph()
    edges_list = []
    nodes_list = []
    for result in results:
        edges = result["edges"]
        nodes = result["nodes"]
        edges_list+=edges
        nodes_list+=nodes
    for node in nodes_list:
        graph.add_node(node)
    for edge in edges_list:
        graph.add_edge(edge[0], edge[1])    
    return graph

def graph_analysis(graph):
    analyzer = CentralityAnalyzer(graph)
    detector = CommunityDetector(graph)
    communities = detector.louvain()
    metrics = GraphMetrics(graph)
    return {
        "pagerank": analyzer.pagerank(),
        "betweenness": analyzer.betweenness(),
        "communities": communities,
        "modularity": detector.modularity(communities),
        "graph_metrics": metrics.summary()
    }
if __name__=="__main__":
    list_of_sentences = [
        "Mathematics is the study of numbers and quantities.",
        "Algebra is an important branch of mathematics.",
        "Geometry deals with shapes and spatial relationships.",
        "Calculus studies change and motion.",
        "Mathematical equations are used to solve problems.",
        "English is a language used for communication.",
        "English grammar describes the structure of sentences.",
        "Vocabulary is important when learning English.",
        "English literature includes many famous books.",
        "Reading improves English language skills.",
        "Python is a popular programming language.",
        "Python is commonly used for artificial intelligence.",
        "Machine learning is a field of artificial intelligence.",
        "Neural networks are used in machine learning.",
        "Python has many libraries for machine learning."
    ]
    topic_name_column = 'Representation'
    graph = create_knowledge_graph(list_of_sentences, topic_name_column)
    results = graph_analysis(graph)
    pprint(results)
