
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
import warnings


import pandas as pd

from mpire import WorkerPool

def get_graphs(N,opacity):
    rv = go.Mesh3d(
        x=(60*np.random.randn(N)),
        y=(25*np.random.randn(N)),
        z=(40*np.random.randn(N)),
        opacity=opacity,
    )
    return rv
    
def get_lists(cols):
    rv = []
    for j in range(cols):
        rv.append({"is_3d":True})
    return rv

def create_crystals(num_of_crystals,opacity,Type):
    
    TypeList = ["2d","3d"]
    print(f"only these types are present:- {TypeList}")
        
    if Type not in TypeList:
        return
    rows = num_of_crystals
    N = 3
    if Type=="3d":
        N = 5
    progress_bar = False
    rows = rows
    cols = rows
    specs_list = []
    num_cores = max(multiprocessing.cpu_count()//2,1)
    results = []
    for i in range(rows):
        my_dict = {
            "cols":cols
        }
        results.append(my_dict)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(get_lists, results, progress_bar=progress_bar)

    for r in results:
        tmp = r
        specs_list.append(tmp)

    fig = make_subplots(rows=rows, cols=cols,
                        specs = specs_list,
                        print_grid=False)

    results = []
    for i in range(1,rows+1):
        my_dict = {
            "N":N,
            "opacity":opacity,
        }
        results.append(my_dict)
    num_cores = max(multiprocessing.cpu_count()-1,1)
    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(get_graphs, results, progress_bar=True)
    fig.add_traces(results)
    rows = rows
    fig.update_layout( width=300*rows,height=300*rows, margin=dict(r=10, l=10, b=10, t=10))
    fig.update_layout(scene_aspectmode='cube')
    fig.update_layout(scene2_aspectmode='manual',
                    scene2_aspectratio=dict(x=1, y=1, z=2))
    fig.update_layout(scene3_aspectmode='data')
    fig.update_layout(scene4_aspectmode='auto')
    return fig

def create_fig(fig):
    fig.show()

def create_main_crystals(num_of_crystals,opacity,Type):
    fig = create_crystals(num_of_crystals, opacity, Type)
    if fig!=None:
        return True,fig
    return False,None
    

if __name__ == "__main__":    
    num_of_crystals = 5
    Type = "3d"
    opacity = 1
    check,fig = create_main_crystals(num_of_crystals,opacity,Type)
    if check==True:
        fig.show()
    
