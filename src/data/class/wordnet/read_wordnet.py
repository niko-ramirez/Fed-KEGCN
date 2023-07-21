import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import pickle as pkl
import os
import sys
from collections import Counter
import random

train_data = np.load("train_data.npz", allow_pickle=True)
test_data = np.load("test_data.npz")
print(train_data.files)
labels = train_data["labels"]
print (labels)