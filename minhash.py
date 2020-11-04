"""
Calculate the signature of each document using minhashing 
technique

functions in module:
    * generate_signature_matrix - get signature matrix from incidence matrix
"""

from random import randint
from pandas import DataFrame, read_pickle
import numpy as np
import os
from tqdm import tqdm


def generate_hash_functions(rows, no_of_hash_func=200):
    """Generates parameters for given no of hash functions
    
    Parameters
    ----------
    rows: int
    no_of_hash_funct: int, optional
        
    Returns
    -------
    list
        list of functions which can be used as hashes[i](x)
    """

    hashes = []
    c = rows
    
    # all functions are same here. check this
    for i in range(no_of_hash_func):
        def hash(x):
            """
            This function calculates hash for given x

            hash function format: (a*x+b)%c where
                c: prime integer just greater than rows
                a,b: random integer less than c
            """
            return (randint(1,5*c)*x + randint(1,5*c))%c
        hashes.append(hash)

    return hashes


def generate_signature_matrix(incidence_matrix, no_of_hash_func=200):
    """Generates the signature matrix for whole corpus

    If sig_mat.pickle exists,it will be used instead

    Parameters
    ----------
    incidence_matrix: pandas.DataFrame
    no_of_hash_func: int, optional
        
    Returns
    -------
    pandas.DataFrame
        dataframe containing signatures of each document
    """

    if os.path.exists("sig_mat.pickle"):
        signature_matrix = read_pickle("sig_mat.pickle")
        print("Using already existing sig_mat.pickle file")
        return signature_matrix

    rows, cols = incidence_matrix.shape
    hashes = generate_hash_functions(rows, no_of_hash_func)
    signature_matrix = DataFrame(index=[i for i in range(no_of_hash_func)], columns=incidence_matrix.columns)
    
    # minhashing algorithm
    for i in tqdm(range(rows)):
        for j in incidence_matrix.columns:
            if incidence_matrix.iat[i,j]==1:
                for k in range(no_of_hash_func):
                    if np.isnan(signature_matrix.iat[k,j]):
                        signature_matrix.iat[k,j] = hashes[k](i)
                    else:
                        signature_matrix.iat[k,j] = min(signature_matrix.iat[k,j], hashes[k](i))
    
    print("signature_matrix is being saved to pickle file.......")
    signature_matrix.to_pickle("sig_mat.pickle")
    print("saved to sig_mat.pickle")
    return signature_matrix