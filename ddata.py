import numpy
import torch
from torch.utils.data import DataLoader, Dataset


class DDataManager:
    def __init__(self, dataset, split=[0.7, 0.1, 0.2], augment=False):
        if isinstance(dataset, list):
            if len(dataset)  !=3:
                raise Exception(f"Error, received {len(dataset)} datasetes, cannot auto-assign train, val, test!")
            print("Found 3 dataset, auto assigning to Train, Val, Test")
            self.train_set = dataset[0]
            self.val_set = dataset[1]
            self.test_set = dataset[2]
        else:
            self.full_set = dataset
            if split is None:
                raise Exception("Error, with singular dataset, split must be set as a list of 3 numbers!")
            if len(split) != 3:
                raise Exception("Error, split must be a list of 3 elements that sum up to 1!")
            if sum(split) != 1:
                raise Exception("Error, split must be a list of 3 elements that sum up to 1!")
                
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(self.full_set, split)
    
    def augment(self):
        print("Default augment method found, so nothing is done.\n- You must overload the augment function in your DDataManager class to provide such functionality!")
        
    def n_features(self):
        f, _ = self.train_set[0]
        return f.numel()
    
    def feature_shape(self):
        f, _ = self.train_set[0]
        return f.size()
    
    def n_features(self):
        _, l = self.train_set[0]
        return l.numel()
    
    def label_shape(self):
        _, l = self.train_set[0]
        return l.size()