import os
import sys
import wandb
print(wandb.__version__)
wandb.login()



import shutil
import datetime
import math
import opendatasets as od
import pandas as pd
from pandas import Series, DataFrame
import string
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ModelArchitecture.CharRNN import CharRNNCustom
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time


#Make a function that finds which plot summaries mention authors, then filter out all in dataset that does this
def mentions_author(row):
    author = str(row["Author"]).lower()
    summary = str(row["Plot_Sum"]).lower()
    
    names = author.split()
    
    first= names[0]
    last = names[-1]
    
    return first in summary or last in summary

def mentions_title(row):
    title = str(row["Book_Title"]).lower()
    summary = str(row["Plot_Sum"]).lower()
    
    
    return title in summary





def seq_to_indices(seq, char_to_idx):
    return torch.tensor([char_to_idx[c] for c in seq])


def make_input_target(seq, char_to_idx):

    indices = seq_to_indices(seq, char_to_idx)

    input_idx  = indices[:-1]
    target_idx = indices[1:]


    return input_idx, target_idx

class CharDataset(Dataset):

    def __init__(self, windows, char_to_idx):
        self.windows = windows
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):

        seq = self.windows[idx]

        x, y = make_input_target(seq, self.char_to_idx)
        return x, y

def is_valid(text):
    allowed_chars = set(
        string.ascii_letters +   
        string.digits +          
        string.punctuation +     
        " \n\t"                 
    )
    return all(c in allowed_chars for c in text)

def split_column_into_train_val_test(df, column_name, train_frac=0.8, val_frac=0.2, random_state=42):
    """
    Splits a specific DataFrame column into training, validation, and test datasets.
    """
    # Ensure fractions sum to 1.0
    if abs(train_frac + val_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0")

    # Extract the specific column as a Series or DataFrame (if keeping single column as DataFrame)
    # Using double brackets [[]] keeps it as a DataFrame, which is often useful for consistency
    data_col = df[[column_name]]

    # Step 1: Split into training and validation sets
    # The test_size for this step is val_frac + test_frac
    X_train, X_val = train_test_split(
        data_col,
        test_size=(val_frac),
        random_state=random_state,
        shuffle=True
    )

    

    return X_train, X_val

def slide_window(data: DataFrame, window_size = 256, stride_size = 64):
    windowed_list = []
    total_samples = 0
    for sample_num in tqdm(range(len(data))):
        i = 0
        window_sample = []
        current_sample = data.iloc[sample_num]["Plot_Sum"]
        #current_sample = data[sample_num]
        sample_done = False
        
        while sample_done is False:
            if((i*stride_size + window_size) < len(current_sample)):
                window_sample = current_sample[i*stride_size:(i*stride_size + window_size)]
                #window_sample = current_sample[(len(current_sample) - window_size):]
                if(len(window_sample) != window_size):
                    print("For full sample",len(window_sample))
            else:
                
                window_sample = current_sample[(len(current_sample) - window_size):]
                if(len(window_sample) != window_size):
                    print("For leftover sample",len(window_sample))
                sample_done = True
            windowed_list.append(window_sample)
            i+=1
            total_samples+=1
    print(f"{total_samples} windowed samples of character length {window_size} made for this data")
    return windowed_list

def generate_text(model, char_to_idx, idx_to_char, max_length=2000, start_text = "<", stop_char = ">", temperature=0.8, device="cpu"):
    
    model.eval()
    
    input_seq = torch.tensor([[char_to_idx[c] for c in start_text]], device=device)
    hidden = model.init_hidden(1, device=device)
    
    
    input_seq = torch.tensor([[char_to_idx[c] for c in start_text]], device=device)  # (1, seq_len)
    hidden = model.init_hidden(1, device=device)

    # Prime the model
    for i in range(len(start_text) - 1):
        _, hidden = model.forward2(input_seq[:, i:i+1], hidden)  # slice instead of index to keep 2D

    current_char = input_seq[:, -1:]  # (1, 1) instead of (1,)

    generated = start_text
    for _ in range(max_length):
        output, hidden = model.forward2(current_char, hidden)

        logits = output.squeeze() / temperature
        probs = torch.softmax(logits, dim=0)

        next_idx = torch.multinomial(probs, 1).item()
        next_char = idx_to_char[next_idx]

        generated += next_char
        if next_char == stop_char:
            break
        current_char = torch.tensor([[next_idx]], device=device)  # (1, 1) instead of (1,)
    """
    # Prime the model with the seed text
    for i in range(len(start_text) - 1):
        _, hidden = model.forward2(input_seq[:, i], hidden)

    current_char = input_seq[:, -1]

    generated = start_text
    for _ in range(max_length):

        output, hidden = model.forward2(current_char, hidden)

        logits = output.squeeze() / temperature
        probs = torch.softmax(logits, dim=0)

        next_idx = torch.multinomial(probs, 1).item()
        next_char = idx_to_char[next_idx]

        generated += next_char
        if(next_char == stop_char):
            break;
        current_char = torch.tensor([next_idx], device=device)

    """
    return generated

#import torch
#import numpy as np


def custom_rnn_training(train_dataset, val_dataset, gru_hidden_list, lstm_hidden_size, epochs, batch_size, learning_rate, vocab_size, embedding_size, window_size, loss_func, save_dir, char_to_idx, idx_to_char, device, early_stopping_patience = 5, trial_num = 0, timestamp="null", optimizer="adam"):
    
    model_time = f"custom_trial_{trial_num}_{timestamp}"
    model_dir = os.path.join(save_dir, model_time)
    if(os.path.isdir(model_dir) is False):
        os.makedirs(model_dir)
    

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="etmorales-uc-san-diego",
        settings=wandb.Settings(init_timeout=360, mode="online"),
        # Set the wandb project where this run will be logged.
        project="Char-RNN-training",
        # Track hyperparameters and run metadata.
        name= model_time,
        
        config={
            "architecture": f"Char-RNN-Custom",
            "dataset": "CMU_BOOK_Summary_Dataset",
            "epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
            "batch_size": batch_size,
            "loss_func" : "CrossEntropyLoss",
            "optimizer" : optimizer,
            "learning_rate": learning_rate,
            "gru_hidden_size_list": gru_hidden_list,
            "lstm_hidden_size": lstm_hidden_size,
        }

        
        )

    train_sample_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_sample_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"For training, there are {len(train_sample_loader)} batches of {batch_size} sequences of character length {window_size}")
    print(f"For validation, there are {len(val_sample_loader)} batches of {batch_size} sequences of character length {window_size}")
    
    net = CharRNNCustom(input_size = vocab_size, gru_hidden_sizes=gru_hidden_list, lstm_hidden_size=lstm_hidden_size, output_size = vocab_size, embedding_size= embedding_size)
    net = net.to(device)

    if(optimizer=="adam"):
        opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)


    train_loss = 0
    train_loss_list = []
    train_accuracy_list = []

    val_loss = 0
    val_loss_list = []
    val_accuracy_list = []

    best_val_loss = torch.inf

    
    
    
    
    best_epoch = -1



    prev_loss = torch.inf
    global_step = 0
    patience = 0
    for epoch in range(epochs):
        #Train portion
        net.train()
        total_train_loss = 0
        total_train_tokens = 0
        train_pbar = tqdm(train_sample_loader, desc = f"Train Progress for Epoch {epoch+1}")
        for batch in train_pbar:
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            batch_size = x.shape[0]
            
            opt.zero_grad()
            hidden = net.init_hidden(batch_size, device)
            output, hidden = net(x, hidden)
            #print("Output Shape", output.shape)
            #print("Target Shape", y.shape)
            output = output.reshape(-1, vocab_size)
            target = y.reshape(-1)

            train_loss = loss_func(output, target)
            train_loss.backward()             # Backward. 
            
            #Track gradient norms
            total_norm = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            #Clip gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            
            opt.step()

            if global_step % 50 == 0:
                wandb.log({"grad_norm": total_norm}, step=global_step)
            global_step += 1

            total_train_loss += train_loss.item() * target.numel()
            total_train_tokens += target.numel()
            prev_loss = train_loss.item()
            train_pbar.set_postfix(last_batch_loss=f"{train_loss:.4f}")
            
        train_pbar.close()
        scheduler.step()
        avg_train_loss = total_train_loss / total_train_tokens
        train_perplexity = math.exp(avg_train_loss)
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "train_perplexity": train_perplexity,
        })


        print(f"In epoch {epoch+1}, the average training loss is {avg_train_loss}")
        train_loss_list.append(avg_train_loss)
        


        net.eval()
        total_val_loss = 0
        total_val_tokens = 0
        val_perplexity = torch.inf
        with torch.no_grad():
            val_pbar = tqdm(val_sample_loader, desc=f"Validation Progress for Epoch {epoch+1}")
            
            for batch in val_pbar:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                batch_size = x.shape[0]

                hidden = net.init_hidden(batch_size, device)
                output, hidden = net(x, hidden)

                output = output.reshape(-1, vocab_size)
                target = y.reshape(-1)

                val_loss = loss_func(output, target)

                total_val_loss += val_loss.item() * target.numel()
                total_val_tokens += target.numel()
                val_pbar.set_postfix(last_batch_loss=f"{val_loss:.4f}")

            val_pbar.close()

        avg_val_loss = total_val_loss / total_val_tokens
        val_perplexity = math.exp(avg_val_loss)
        

        wandb.log({
            "epoch": epoch + 1,
            "avg_val_loss": avg_val_loss,
            "val_perplexity": val_perplexity,
        })
        print(f"In epoch {epoch+1}, the average validation loss is {avg_val_loss}")
        val_loss_list.append(avg_val_loss)
        val_accuracy_list.append(val_perplexity)


        if(avg_val_loss < best_val_loss):
            print(f"New best val loss for , printing new model weights file and showing generated text for epoch {epoch+1}")
            generated_text = generate_text(net, char_to_idx=char_to_idx, idx_to_char=idx_to_char, device=device)
            print(f"Generated Text:\n{generated_text}")
            file_name = f"custom_rnn_trial_{trial_num}_epoch_{epoch+1}_{timestamp}.pt"
            weights_path = os.path.join(model_dir, file_name)
            torch.save(net.state_dict(), weights_path)
            best_val_loss = avg_val_loss
            best_epoch=epoch+1
            patience=0
        else:
            patience+=1
            if(patience == early_stopping_patience):
                print(f"Model hasn't improved within early stopping patience of {early_stopping_patience} epochs. Stopping model training")
                break;
            
    run.finish()
    del net
    torch.cuda.empty_cache()
    return best_epoch, best_val_loss

def main():
    CHAR_LEN = 256
    

    save_dir = "model_weights"

    od.download('https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset')
    columns = ["WikiID", "FreebaseID", "Book_Title","Author","Pub_date", "Book_Genre","Plot_Sum"]
    book_summary_dataset = pd.read_csv("./cmu-book-summary-dataset/booksummaries.txt", sep="\t", header=None, names = columns)
    print(f"There are {book_summary_dataset.shape[0]} book summaries in the dataset initially")

    pure_story_summary = book_summary_dataset[~book_summary_dataset.apply(mentions_author, axis=1)]
    pure_story_summary = pure_story_summary[~pure_story_summary.apply(mentions_title, axis=1)]
    
    pure_story_summary = pure_story_summary[pure_story_summary["Plot_Sum"].apply(is_valid)]
    pure_story_summary = pure_story_summary[~pure_story_summary["Plot_Sum"].str.contains(r"[<>]", regex=True, na=False)]

    pure_story_summary["Plot_Sum"] = pure_story_summary["Plot_Sum"].str.strip()
    pure_story_summary["Plot_Sum"] = "<" + pure_story_summary["Plot_Sum"] + ">"
    pure_story_summary["Sum_Char_Count"] = pure_story_summary["Plot_Sum"].str.len()


    pure_story_summary = pure_story_summary[pure_story_summary['Plot_Sum'].str.len() >= CHAR_LEN]
    single_string = pure_story_summary['Plot_Sum'].str.cat(sep='')

    chars = sorted(set(single_string))
    #all_text = "".join(pure_story_summary["Plot_Sum"])
    vocab_size = len(chars)

    char_to_idx = {c:i for i,c in enumerate(chars)}
    idx_to_char = {i:c for i,c in enumerate(chars)}
    print(len(chars))
    print(f"After filtering out all summaries that include the authors name, title name, and that might have <> symbols in them we have {pure_story_summary.shape[0]} book summaries in the dataset")
    total_char = 0
    avg_char = 0

    total_char = pure_story_summary["Sum_Char_Count"].sum()

    avg_char = total_char / pure_story_summary.shape[0]
    print(f"Total characters is {total_char}")
    print(f"Average characters is {round(avg_char)}")

    X_train, X_val = split_column_into_train_val_test(pure_story_summary, "Plot_Sum")
    print(f"Training data has {X_train.shape[0]} summaries and validation data has {X_val.shape[0]} summaries")

    window_size = 256
    stride_size = 64

    X_train_windowed = slide_window(X_train, window_size=window_size, stride_size=stride_size)
    X_val_windowed = slide_window(X_val, window_size=window_size, stride_size=stride_size)

    print(f"The train dataset when put through a sliding window of the previous parameters has {len(X_train_windowed) * window_size} characters in total")
    print(f"The val dataset when put through a sliding window of the previous parameters has {len(X_val_windowed) * window_size}")

    #Simple hyperparameter choices for grid search
    batch_sizes = [128, 256]
    epochs = [20, 30]
    early_stopping_patiences = [5, 10] #Either eary stopping patience of 5 or 10
    learning_rates = [5e-4, 1e-3]
    gru_hidden_sizes = [[64, 128, 256], [128, 256, 512]]
    lstm_hidden_sizes = [512, 1024]
    embedding_size = 128
    trial_amount = len(batch_sizes)*len(epochs)*len(early_stopping_patiences)*len(learning_rates)*len(gru_hidden_sizes)*len(lstm_hidden_sizes)
    
    loss_func = nn.CrossEntropyLoss() 
    opt = "adam"
    
    train_dataset = CharDataset(X_train_windowed, char_to_idx)
    val_dataset = CharDataset(X_val_windowed, char_to_idx)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Using device:", device) 
    #Grid Search
    print(f"Total trials is {trial_amount}")
    current_trial = 1
    best_custom_val_loss = torch.inf
    best_hyperparameter_setup = {}
    for early_stopping_patience in early_stopping_patiences:
        for epoch_amount in epochs:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for gru_hidden_size in gru_hidden_sizes:
                        for lstm_hidden_size in lstm_hidden_sizes:
                            #Doing this as I needed to change a specific part of the code
                            if(current_trial < 9):
                                print(f"Skipping trial {current_trial} as it has already been done")
                                current_trial+=1
                                continue;
                            print(f"Trial Number {current_trial} of {trial_amount} is beginning")
                            best_trial_epoch, best_trial_val_loss = custom_rnn_training(train_dataset=train_dataset, val_dataset=val_dataset, gru_hidden_list=gru_hidden_size, lstm_hidden_size=lstm_hidden_size, epochs=epoch_amount, early_stopping_patience=early_stopping_patience, batch_size=batch_size, learning_rate=learning_rate, vocab_size=vocab_size, embedding_size=embedding_size, window_size=window_size, loss_func=loss_func, save_dir=save_dir, char_to_idx=char_to_idx, idx_to_char=idx_to_char, device=device, timestamp=timestamp, trial_num=current_trial)
                            if(best_trial_val_loss < best_custom_val_loss):
                                print(f"Trial {current_trial} is the new best trial")
                                best_custom_val_loss = best_trial_val_loss
                                best_hyperparameter_setup = {
                                    "trial": current_trial,
                                    "best_epoch": best_trial_epoch,
                                    "best_val_loss": best_trial_val_loss,
                                    "batch_size": batch_size,
                                    "epochs": epoch_amount,
                                    "early_stopping_patience": early_stopping_patience,
                                    "learning_rate": learning_rate,
                                    "gru_hidden_sizes": gru_hidden_size,
                                    "lstm_hidden_size": lstm_hidden_size,
                                }
                            current_trial+=1
                            if(current_trial % 5 == 0):
                                #Give GPU a second to cool
                                print("4 trials have been completed in succession, letting GPU cool for 5 minutes")
                                time.sleep(300)
 
    print(f"BEST CUSTOM RNN LOSS: {best_custom_val_loss} ")
    print(f"BEST HYPERPARAMTER SETUP:")
    for key, value in best_hyperparameter_setup.items():
        print(f"Key: {key}, Value: {value}")



def test():
    CHAR_LEN = 256
    

    save_dir = "model_weights"

    od.download('https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset')
    columns = ["WikiID", "FreebaseID", "Book_Title","Author","Pub_date", "Book_Genre","Plot_Sum"]
    book_summary_dataset = pd.read_csv("./cmu-book-summary-dataset/booksummaries.txt", sep="\t", header=None, names = columns)
    print(f"There are {book_summary_dataset.shape[0]} book summaries in the dataset initially")

    pure_story_summary = book_summary_dataset[~book_summary_dataset.apply(mentions_author, axis=1)]
    pure_story_summary = pure_story_summary[~pure_story_summary.apply(mentions_title, axis=1)]
    
    pure_story_summary = pure_story_summary[pure_story_summary["Plot_Sum"].apply(is_valid)]
    pure_story_summary = pure_story_summary[~pure_story_summary["Plot_Sum"].str.contains(r"[<>]", regex=True, na=False)]

    pure_story_summary["Plot_Sum"] = pure_story_summary["Plot_Sum"].str.strip()
    pure_story_summary["Plot_Sum"] = "<" + pure_story_summary["Plot_Sum"] + ">"
    pure_story_summary["Sum_Char_Count"] = pure_story_summary["Plot_Sum"].str.len()


    pure_story_summary = pure_story_summary[pure_story_summary['Plot_Sum'].str.len() >= CHAR_LEN]
    single_string = pure_story_summary['Plot_Sum'].str.cat(sep='')

    chars = sorted(set(single_string))
    all_text = "".join(pure_story_summary["Plot_Sum"])
    vocab_size = len(chars)

    char_to_idx = {c:i for i,c in enumerate(chars)}
    idx_to_char = {i:c for i,c in enumerate(chars)}
    print(len(chars))
    print(f"After filtering out all summaries that include the authors name, title name, and that might have <> symbols in them we have {pure_story_summary.shape[0]} book summaries in the dataset")
    total_char = 0
    avg_char = 0

    total_char = pure_story_summary["Sum_Char_Count"].sum()

    avg_char = total_char / pure_story_summary.shape[0]
    print(f"Total characters is {total_char}")
    print(f"Average characters is {round(avg_char)}")

    X_train, X_val = split_column_into_train_val_test(pure_story_summary, "Plot_Sum")
    print(f"Training data has {X_train.shape[0]} summaries and validation data has {X_val.shape[0]} summaries")

    window_size = 256
    stride_size = 64

    X_train_windowed = slide_window(X_train, window_size=window_size, stride_size=stride_size)
    X_val_windowed = slide_window(X_val, window_size=window_size, stride_size=stride_size)

    print(f"The train dataset when put through a sliding window of the previous parameters has {len(X_train_windowed) * window_size} characters in total")
    print(f"The val dataset when put through a sliding window of the previous parameters has {len(X_val_windowed) * window_size}")

    #Simple hyperparameter choices for grid search
    batch_size = 256
    epoch_amount = 30
    early_stopping_patience = 10 #Either eary stopping patience of 5 or 10
    learning_rate = 1e-3
    gru_hidden_size = [128, 256, 512]
    lstm_hidden_size = 1024
    embedding_size = 128
    
    
    loss_func = nn.CrossEntropyLoss() 
    opt = "adam"
    
    train_dataset = CharDataset(X_train_windowed, char_to_idx)
    val_dataset = CharDataset(X_val_windowed, char_to_idx)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    print("Using device:", device) 

    best_trial_epoch, best_trial_val_loss = custom_rnn_training(train_dataset=train_dataset, val_dataset=val_dataset, gru_hidden_list=gru_hidden_size, lstm_hidden_size=lstm_hidden_size, epochs=epoch_amount, early_stopping_patience=early_stopping_patience, batch_size=batch_size, learning_rate=learning_rate, vocab_size=vocab_size, embedding_size=embedding_size, window_size=window_size, loss_func=loss_func, save_dir=save_dir, char_to_idx=char_to_idx, idx_to_char=idx_to_char, device=device)
    print("DONE")

if __name__ == "__main__":
    main()


