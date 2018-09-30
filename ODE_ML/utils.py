import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from time import time
import sys
import io
import pickle



def pickle_dump(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

    
def pickle_load(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data 

def generate_samples(n, cut, N, d, flag=False):
    data = []  
    for k in range(0, N):
        p = np.random.rand()
        if p<=cut:
            spectre = np.diag((np.random.rand((n)) - 1)*d/2)
            label = 1
        else:
            spectre = np.diag((np.random.rand((n)) - 1/2)*d)
            label = 0   
        new_basis = np.random.rand(n, n)
        system = np.linalg.inv(new_basis).dot(spectre.dot(new_basis))
        temp_list = list(system.reshape(1, -1).flatten()) 
        if flag==True:
            temp_list += list(np.linalg.eigvals(system))
            # temp_list.append(np.max(np.linalg.eigvals(system)))
        temp_list.append(label)
        data.append(temp_list)

    data = np.matrix(data)
    X = np.asarray(data[:, :-1])
    Y = np.asarray(data[:, -1]).flatten().astype('int')
    return X, Y