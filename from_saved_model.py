#This will be a script that loads our model and runs evaluations

#from datasets import load_dataset
#from transformers import AutoTokenizer
#import numpy as np
#import evaluate
import torch
#from transformers import Trainer
#from torch.utils.data import DataLoader
#from transformers import AutoModelForSequenceClassification
#from torch.optim import AdamW
#from transformers import get_scheduler
#from tqdm.auto import tqdm
#import evaluate




if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("cpu")
