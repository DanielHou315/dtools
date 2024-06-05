import os
import re

import numpy as np
import torch
from torch import nn
from torchinfo import summary

from tqdm import tqdm
import matplotlib.pyplot as plt

from .dtrainer_utils import get_best_device



class DTrainer():    
    def __init__(self
                 , identity              # Identify model, also give name to 
                 , model                 # The actual model
                 , dataset               # The torch.utils.data.Dataset derived clas object
                 , train_loader          # torch.utils.data.DataLoader for training data
                 , val_loader
                 , test_loader
                 , criterion             # Loss function
                 , optimizer             # SGD, Adam, etc
                 , scheduler             # How learning rate should change
                 , ckpt_path             # Place to store checkpoints
                 , result_path           # Place to store outputs
                 , save_cycle=5
                 , transferred=False
                 , benchmarking=False
                 , device=None):
        if not isinstance(identity,str):
            raise Exception("[DTrainer] Error! Identity for DTrainer is not a string!")
            
        self.identity = identity
        self.device = get_best_device(device)
        self.model = model
        
        print("[DTrainer] Loaded Model", self.model)
        
        # Data loading
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        
        print("[DTrainer] Loaded Data Loaders with input shape:", self.get_train_shape()[0], \
              "output_shape:", self.get_train_shape()[1])
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ckpt_path = ckpt_path
        self.result_path = result_path
        
        self.graph_path = os.path.join(result_path, "graphs")
        if not os.path.isdir(self.graph_path):
            os.makedirs(self.graph_path)
        self.output_path = os.path.join(result_path, "output")
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        
        self.current_epoch = 1
        self.save_cycle = save_cycle
        
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.transferred = transferred
        
        self.benchmarking = benchmarking
        self.val_benchmark_results = []
        
        # input_size = (self.train_loader.batch_size, )
        # input_size += self.dataset[0][0].size()
        # print(input_size)
        # summary(self.model, input_size=(40,9))
        
        print("[DTrainer] Trainer Initialized")
        
        existing_epochs = self.extract_epochs()
        print("[DTrainer] Found checkpoint backup for these epochs:", existing_epochs)
        
    def get_train_shape(self):
        for i, (f, l) in enumerate(self.train_loader):
            return (f.size(),l.size())

    def train_epoch(self, train_loader):
        """Train the `model` for one epoch of data from `data_loader`.

        Use `optimizer` to optimize the specified loss
        """
        train_loss = 0
        self.model = self.model.to(self.device)
        for i, (X, y) in enumerate(self.train_loader):
            X,y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            
        if self.scheduler is not None: 
            self.scheduler.step()
        train_loss /= len(self.train_loader)
        self.train_losses.append(train_loss)
        return train_loss

    
    def val_epoch(self):
        """Validate each epoch"""
        val_loss = 0
        self.model = self.model.to(self.device)
        with torch.no_grad():
            for features, labels in self.val_loader:
                output = self.model(features.to(self.device))
                val_loss += self.criterion(output, labels.to(self.device)).item()
            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)
            return val_loss

    def train(self, start_epoch=-1, num_epoch=10, force_retrain=False):
        """Train model"""
        if start_epoch == -1:
            start_epoch = self.current_epoch
            
        if start_epoch != self.current_epoch:
            if not force_retrain:
                print("Exist checkpoints for epochs", self.extract_epochs())
                try:
                    self.restore_checkpoint(start_epoch)
                except:
                    raise Exception(f"Checkpoint of Epoch {start_epoch} does not exist!")
            else:
                self.current_epoch = 1
                start_epoch = 1
                print("Retraining from epoch 1!")
        for epoch in range(start_epoch, start_epoch+num_epoch):
            self.current_epoch = epoch
            
            # Train model
            with tqdm(self.train_loader) as pbar:
                train_loss = self.train_epoch(pbar)

                # Save Checkpoints and show loss
                if (self.current_epoch) % self.save_cycle == 0:
                    if self.ckpt_path is not None:
                        self.save_checkpoint()
                    else:
                        print("Warning! Checkpoint not saved due to None checkpoint_path!")
                    self.show_losses()

                # Validation
                val_loss = self.val_epoch()

                # Summarize
                print(f"Epoch {epoch}, Training Loss: {train_loss}, Val Loss: {val_loss}")
        return (train_loss, val_loss)
            
    def test(self):
        """Test model on test loader"""
        test_loss = 0
        self.model = self.model.to(self.device)
        with torch.no_grad():
            for features, labels in self.test_loader:
                output = self.model(features.to(self.device))
                test_loss += self.criterion(output, labels.to(device)).item()
            test_loss /= len(test_loader)
            print("Test Loss:", test_loss)
            self.show_benchmark()
            return test_loss
            
    def get_checkpoint_name(self, epoch):
        """Formatted as <identity>-ep<epoch-number>.checkpoint.pth.tar"""
        return f"{self.identity}-ep-{epoch}.checkpoint.pth.tar"
    
    def save_checkpoint(self):
        PATH = os.path.join(self.ckpt_path, self.get_checkpoint_name(self.current_epoch))
        torch.save({
                'identity': self.identity,
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler,
                'checkpoint_path': self.ckpt_path,
                'result_path': self.result_path,
                'train_loss': self.train_losses,
                'val_loss': self.val_losses,
                'test_loss': self.test_losses
                }, PATH)
        print(f"Model {self.identity} from epoch {self.current_epoch} saved")
        
    def extract_epochs(self):
        directory = self.ckpt_path
        # pattern to capture the epoch number from the file name
        epoch_pattern = re.compile(r"-ep-(\d+).checkpoint.pth.tar$")
        epochs = []

        # list all files in the directory
        for filename in os.listdir(directory):
            # search for the pattern. If it's a match, extract the epoch number
            match = epoch_pattern.search(filename)
            if match:
                # add the captured epoch number to the list
                epochs.append(int(match.group(1)))
        return epochs
    
    def restore_latest_checkpoint(self, force=False):
        epochs = self.extract_epochs()
        if len(epochs) == 0:
            raise Exception("No Epochs found!")
        epoch = epochs[-1]
        self.restore_checkpoint(epoch, force)
        
    def restore_checkpoint(self, _epoch, force=False):
        """
        Restore model from checkpoint if it exists
        Returns the model and the current epoch.
        """
        self.save_checkpoint()    # Always backup current state first
        try:
            cp_files = [
                file_
                for file_ in os.listdir(self.ckpt_path)
                if file_.startswith(self.identity) and file_.endswith(".checkpoint.pth.tar")
            ]
        except FileNotFoundError:
            cp_files = None
            os.makedirs(self.ckpt_path)
        if not cp_files:
            print("No saved model parameters found")
            raise Exception("Checkpoint not found")

        filename = os.path.join(
            self.ckpt_path, f"{self.identity}-ep-{_epoch}.checkpoint.pth.tar"
        )

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location=self.device)
        
        self.current_epoch = checkpoint["epoch"] + 1
        if self.transferred:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.identity = checkpoint["identity"]
            self.scheduler = checkpoint["scheduler"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_loss"]
        self.val_losses = checkpoint["val_loss"]
        self.test_losses = checkpoint["test_loss"]

        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )

    def show_losses(self):
        """
        Show training and validation losses
        """                         
        plt.plot(np.arange(1, len(self.train_losses)+1), np.array(self.train_losses), label="Train Loss")
        plt.plot(np.arange(1, len(self.val_losses)+1), np.array(self.val_losses), label="Val Loss")
        plt.legend()
                                      
        plt.show()
        plt.savefig(os.path.join(self.result_path, "graphs", f"loss_ep{self.current_epoch}.png"))
        plt.close()
        
    
        
                                      
    ## ------------------------
    ## 
    ## METHODS TO BE OVERLOADED
    ## 
    ## ------------------------
    def plot_pred(self, batch):
        print("[DTrainer]: Error! Called base method. You should overwrite this method with your own implementation")
        raise Exception("Virtual Method Called")
        
    def benchmark(self, data_loader):
        print("[DTrainer]: Error! Called base method. You should overwrite this method with your own implementation")
        raise Exception("Virtual Method Called")
    
    def show_benchmark_result(self, data_loader):
        print("[DTrainer]: Error! Called base method. You should overwrite this method with your own implementation")
        raise Exception("Virtual Method Called")

    def pred(self, X):
        """Predict with input X"""
        print("[DTrainer]: Error! Called base method. You should overwrite this method with your own implementation")
        raise Exception("Virtual Method Called")
