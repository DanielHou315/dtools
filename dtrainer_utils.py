import torch

DEV_DEFAULT = 0
DEV_CUSTOM = 1
DEV_ERR = 2
    
def get_best_device(i_device):
    """Get Device to train on"""
    if i_device is None:
        if torch.cuda.is_available():
            return print_rtn_dev('cuda', DEV_DEFAULT)
        if torch.backends.mps.is_available():
            return print_rtn_dev('mps', DEV_DEFAULT)
        return print_rtn_dev('cpu', DEV_DEFAULT)

    # If specified
    if i_device == 'cuda':
        if not torch.cuda.is_available():
            return print_rtn_dev(i_device, DEV_ERR)
        return print_rtn_dev(i_device, DEV_CUSTOM)
    if i_device == 'mps':
        if not torch.backends.mps.is_available():
            return print_rtn_dev(i_device, DEV_ERR)
        return print_rtn_dev(i_device, DEV_CUSTOM)
    if i_device == 'cpu':
        return print_rtn_dev(i_device, DEV_CUSTOM)
    return print_rtn_dev(i_device, DEV_ERR)


def print_rtn_dev(dev, status=DEV_DEFAULT):
    """Helper Function"""
    if status == DEV_DEFAULT:    # Default device
        print(f"[DTrainer] Default to device {dev}")
        return torch.device(dev)
    if status == DEV_CUSTOM:
        print(f"[DTrainer] Customized device {dev}")
        return torch.device(dev)
    print(f"[DTrainer] Cannot find device {dev}, defaulting to 'cpu'")
    return torch.device('cpu')



import numpy as np
import csv

def loadCSV(filename):
  table = np.genfromtxt(filename, delimiter=',')
  print("Loaded CSV with shape", table.shape)
  return table

def loadTSV(filename):
  table = np.genfromtxt(filename, delimiter='\t')
  print("Loaded TSV data with shape", table.shape)
  return table

def loadNPY(filename):
  table = np.load(filename)
  print("Loaded NPY data with shape", table.shape)
  return table

def cleanTable(table):
  cleaned_table = table[~np.isnan(table).any(axis=1)]

  print(f"Retained {cleaned_table.shape[0]} entries from {table.shape[0]}")
  print(f"Removed indices {np.where(np.isnan(table).any(axis=1))}")
  return cleaned_table

def saveTable(table, filename):
  np.save(filename+"_cleaned.npy", table)
  print("File Saved to " + filename+"_cleaned.npy")

def get_header(filename):
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    header = []
    for row in csv_reader:
      list_of_column_names.append(row)
      break
  # Map header name to row
  header_dict = {}
  for i in len(header):
    header_dict[header[i]] = i
  return header_dict
