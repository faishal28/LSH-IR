"""
Build Incidence matrix using shingling process

functions in module:
    * list_files - returns list of files in given directory
    * get_shingle_matrix - returns incidence-matrix
"""

import numpy as np
import codecs
import os
import pickle
from tqdm import tqdm
from pandas import DataFrame,read_pickle
from time import time

def list_files(path, ext=".txt"):
    """returns list of files in given directory

    Parameters
    ----------
    path: string
    ext: string, optional
    
    Returns
    -------
    list
        files in given directory
    """

    print(f"Reading corpus: {path}")
    if(not os.path.exists(path)):
        raise Exception(f"Given folder path: {path} does not exist")
        return
    
    result = []
    i = 0       

    if ext == None:
        ext = ""
    
    for (root, dirs, files) in os.walk(path):
        for f in files:
            if f.endswith(ext):
                result.append( (os.path.join(root, f), i) )
                i += 1
    
    return result

def build_matrix(files, k=4, newline=False):
    """
    builds incidence matrix

    """

    df = DataFrame(columns=[x[1] for x in files])

    for f in tqdm(files):
        with codecs.open(f[0], 'r', encoding="utf8", errors='ignore') as doc:
            data = doc.read()
            data = data.lower()             # lowercase
            data = ' '.join(data.split())   # substitute multiple space with single space
            data = data.replace('\r\n', ' ') # replace windows line endings with space
            data = data.replace('\r', '')   # removes \r in windows
            data = data.replace('\t', '')   # removes tab-spaces
            if newline is False:
                data = data.replace('\n',' ')

            for i in range(0, len(data)-k+1):
                shingle = data[i:i+k]
                if (shingle in df.index) == False:
                    df.loc[shingle] = [0 for i in range(df.shape[1])]
                df.at[ shingle, f[1] ] = 1
    
    return df


def get_shingle_matrix(path, shingle_size=4, ext=".txt"):
    """
    Shingling is performed and incidence index is built

    if pickle file exists, it will will be loaded

    Parameters
    ----------
    path: string
    shingle_size: integer, optional
    ext: string, optional
    
    Returns
    -------
    pandas.Dataframe
        dataframe containing rows as shingles and cols as doc_ids
    """

    incidence_matrix = None
    if os.path.exists(f"{path}_inc_mat.pickle"):
        incidence_matrix = read_pickle(f"{path}_inc_mat.pickle")
        if os.path.exists("file_list.pickle"):
            print(f"Using already created {path}_inc_mat.pickle file")
            print("using pickled file list")
            with open("file_list.pickle", 'rb') as file_list_pkl:
                files = pickle.load(file_list_pkl)
            return incidence_matrix, files
        print("file list not found")

    files = list_files(path, ext)
    incidence_matrix = build_matrix(files, k=shingle_size)

    print("saving generated incidence index to file...")
    incidence_matrix.to_pickle(f"{path}_inc_mat.pickle")
    with open("file_list.pickle", 'wb') as file_list_pkl:
        pickle.dump(files, file_list_pkl)
    print(f"saved to {path}_inc_mat.pickle")
    return incidence_matrix, files