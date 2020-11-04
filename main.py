"""
Program to check document similarity using LSH

Steps:
    - shingling: converting documents to shingles and creating sparse matrix out of it
    
    - minhashing: reducing size of incidence matrix by use of signature
    
    - lsh: determing the documents belonging to same buckets

NOTE: Pickle file is used as cache, if pickle file is present it used instead of creating new incidence matrix
"""

import shingles
import minhash
import lsh
import statistics
import time, os

# creating shingles
start_time = time.time()       
ext =".txt"
path_folder = "corpus"             
shingle_length = 4           
shingle_matrix, files = shingles.get_shingle_matrix(path_folder, shingle_length, ext)
print(shingle_matrix.shape)
print(f"shingling time: {time.time()-start_time}")

# creating minhashes
start_time = time.time()   
no_of_hash_functions = 50   
signature_matrix = minhash.generate_signature_matrix(shingle_matrix, no_of_hash_functions)
print(f"minhashing time: {time.time()-start_time}")

# performing LSH
start_time = time.time()   
r = 2
buckets_list = lsh.get_bucket_list(signature_matrix, r)
print(f"lsh time: {time.time()-start_time}")

# Done with preprocessing, checking similarity between documents
similarity_measure = "cosine"
while True:
    test_doc = input("Path of the test file: ")
    if not os.path.exists(test_doc):
        print("Invalid file path")
        continue
    for name, num in files:
        if test_doc == name:
            test_doc = int(num)

    threshold = float(input("Threshold value: "))

    print(f"Given file: {statistics.get_file_name(test_doc, files)}")
    similar_docs = lsh.find_similar_docs(test_doc, buckets_list, signature_matrix, r)
    output = statistics.compute_similarity(test_doc, similar_docs, shingle_matrix, similarity_measure)

    for file_id, score in output:
        print(f"{statistics.get_file_name(file_id, files)}\t{score}")
    
    print(f"Precision: {statistics.precision(threshold, output)}")
    print(f"Recall: {statistics.recall(threshold, test_doc, len(files), output, shingle_matrix, similarity_measure)}")
