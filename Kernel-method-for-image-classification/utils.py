import matplotlib.pyplot as plt
import numpy as np


def inspect_label(labels):
  unique_elements, counts = np.unique(labels, return_counts=True)
  plt.bar(unique_elements, counts, color='skyblue')
  plt.show()

def scaler(train_set, val_set = None, test_set = None):
    mu_X = train_set.mean()
    sigma_X =train_set.std()
    return (train_set - mu_X)/sigma_X, (val_set - mu_X)/sigma_X, (test_set - mu_X)/sigma_X

def create_Submissioncsv(y_pret):
    f = open("Yte.csv", "w")
    f.write("Id,Prediction\n")
    for n in range(len(y_pret)):
        f.write("{},{}\n".format(int(n+1),y_pret[n]))
    f.close()