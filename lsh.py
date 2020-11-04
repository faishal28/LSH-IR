def band_hashing(band_, hf, buc):
    for colm in band_.columns:
        h = hf(tuple(band_[colm].values))
        if h in buc: 
            buc[h].append(colm)
        else: 
            buc[h] = [colm]


def get_bucket_list(mat, r_, hash_func=None):

    n = mat.shape[0]
    b_ = n//r_
    buckets_list = [dict() for i in range(b_)]

    if hash_func==None:
        hash_func = hash

    for i in range(0, n-r_+1, r_):
        band = mat.loc[i:i+r_-1,:]
        band_hashing(band, hash_func, buckets_list[int(i/r_)])

    return buckets_list


def query_band_hashing(band, hash_func):

    hash_list = []
    h = hash_func(tuple(band.values))
    hash_list.append(h)
    
    return hash_list


def find_similar_docs(doc_id, buckets_list, mat, r, hash_func=None):

    n = mat.shape[0]
    b = n//r

    if hash_func==None:
        hash_func = hash
    
    query_bucket_list = []

    for i in range(0, n-r+1, r):
        band = mat.loc[i:i+r-1, int(doc_id)]
        query_bucket_list.append(query_band_hashing(band, hash_func))
    
    similar_docs = set()
    for i in range(len(query_bucket_list)):
        for j in range(len(query_bucket_list[i])):
            similar_docs.update(set(buckets_list[i][query_bucket_list[i][j]]))

    return similar_docs
