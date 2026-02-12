import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from inari.data import MyDataset_make

num_features = 8 #cox2:35, dd:88, synthetic:8
dataset = MyDataset_make("data/synthetic.pt", num_features)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
targets2 = []
queries2 = []
mms2 = []
for t, q, m in train_set:
    targets2.append(t)
    queries2.append(q)
    mms2.append(m)
torch.save([targets2, queries2, mms2], f"data/synthetic_train.pt")
print("saved trainset") 

targets2 = []
queries2 = []
mms2 = []
for t, q, m in val_set:
    targets2.append(t)
    queries2.append(q)
    mms2.append(m)
torch.save([targets2, queries2, mms2], f"data/synthetic_val.pt")
print("saved valset") 

targets2 = []
queries2 = []
mms2 = []
for t, q, m in test_set:
    targets2.append(t)
    queries2.append(q)
    mms2.append(m)
torch.save([targets2, queries2, mms2], f"data/synthetic_test.pt")
print("saved testset") 
