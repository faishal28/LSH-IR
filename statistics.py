"""
Metrics required to evaluate LSH.
"""
import numpy as np

def jaccard(x, a, signature_matrix):
    """Finds jaccard similarity between two documents

    Parameters
    ----------
    x: int
    a: int
        
    signature_matrix: pandas DataFrame
        contains signature vectors of all documents as columns
    
    Returns
    -------
    int
        jaccard similarity between documents x and a
    """
    x = signature_matrix[x]
    a = signature_matrix[a]
    return sum(x & a)/sum(x | a)


def cosine(x, a, signature_matrix):
    """Finds cosine similarity between two documents

    Parameters
    ----------
    x: int
    a: int
    signature_matrix: pandas DataFrame
        contains signature vectors of all documents as columns
    
    Returns
    -------
    int
        cosine similarity between documents x and a
    """
    x = signature_matrix[x]
    a = signature_matrix[a]
    return np.dot(a,x)/(np.sum(a**2) * np.sum(x**2))**0.5


def compute_similarity(x, similar_docs, signature_matrix, sim_type="jaccard"):
    """Finds similarity between documents

    Parameters
    ----------
    x: int
    similar_docs: list
    signature_matrix: pandas DataFrame
        contains signature vectors of all documents as columns
    sim_type: string
        can take values jaccard, cosine. 

    Returns
    -------
    list
        sorted list of (docid, score) tuples.
    """
    if sim_type == "jaccard": sim_fun = jaccard
    elif sim_type == "cosine": sim_fun = cosine
    # write for all other funcs
    ranked_list = []
    for i in similar_docs:
        if i == x: continue
        score = sim_fun(x, i, signature_matrix)
        ranked_list.append((i, score))
    
    return sorted(ranked_list, key=lambda x: x[1], reverse=True)


def precision(threshold, output):
    """Finds precision

    Parameters
    ----------
    threshold: float
    output: list

    Returns
    -------
    float
        precision value for the given set of retrieved items.
    """
    req = [ i for f, i in output if i>=threshold ]
    try:
        return len(req)/len(output)
    except:
        print("not defined")


def recall(threshold, x, size, output, signature_matrix, sim_type):
    """Finds recall

    Parameters
    ----------
    threshold: float
    x: int
    size: int
    output: list
    signature_matrix: pandas DataFrame
    sim_type: string 

    Returns
    -------
    float
        recall value for the given set of retrieved items.
    """
    docs = compute_similarity(x, [ i for i in range(size) ], signature_matrix, sim_type)
    req = [ i for f, i in output if i>=threshold ]
    den = [ i for f, i in docs if i>=threshold and f!=x ]
    if len(den) == 0:
        return "not defined"
    return len(req)/len(den)


def get_file_name(file_id, files):
    """Get file name

    Parameters
    ----------
    threshold: float
    files: list

    Returns
    -------
    string
        name of the file with given file_id.
    """
    for filename, f_id in files:
        if file_id == f_id:
            return filename