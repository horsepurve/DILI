''' 
This is code can be used for:
    1. run training
    2. run generation
    3. run generation for different checkpoints
    4. run generation for CAMDA val sets 
    5. run generation for visualization
'''
import os
import sys
from random import randrange
import torch
from functools import partial
import time
import pandas as pd
import numpy as np 
import hashlib
import copy

from datasets import load_dataset, Dataset
from llama_help import create_bnb_config, load_model
from llama_help import df_split, lab_dist, parse_max_len
from llama_help import create_prompt_formats 
from llama_help import get_max_length, preprocess_batch, preprocess_dataset
from llama_help import create_peft_config, find_all_linear_names
from llama_help import print_trainable_parameters, fine_tune

''' 
RUN_MODE:
    train      - run training
    test       - test on k1 k2 k3
    checkpoint - look through checkpoints
    allval     - all val sets
    vis        - visualization only
'''
RUN_MODE = 'vis'
def check_mode(run_mode, a_str):
    # first, prevent typo
    assert a_str in ['train','test','checkpoint','allval','vis']
    return (run_mode == a_str)
#%% Initializing Transformers and Bitsandbytes Parameters
##########################################################################
# transformers parameters
##########################################################################

# The pre-trained model from the Hugging Face Hub to load and fine-tune
model_name = "meta-llama/Llama-2-7b-chat-hf"

##########################################################################
# bitsandbytes parameters
##########################################################################

# Activate 4-bit precision base model loading
load_in_4bit = True

# Activate nested quantization for 4-bit base models (double quantization)
bnb_4bit_use_double_quant = True

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Compute data type for 4-bit base models
bnb_4bit_compute_dtype = torch.bfloat16 # bfloat16 | float16

#%% Load model from Hugging Face Hub with model name and bitsandbytes configuration
# ========== ========== 
# llama-2 model
# ========== ==========

bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)

model, tokenizer = load_model(model_name, bnb_config)

#%% Loading Dataset
# ========== ========== 
# Dataset
# ========== ==========

# saved or saved_dili
data_infile = "saved/DILI.csv"  
CH_data = pd.read_csv(data_infile) 

## my_prompt = 'Categorize the research article into one of the 2 categories:\n\nRelevant\nIrrelevant\nNo explanation is needed.\n\n'
if 'DILI' in data_infile:
    my_topic = 'Drug-Induced Liver Injury (DILI)'
else:
    print('!! Data Error !!')
print('~~~~~ Experiment on topic: ', my_topic, '~~~~~')
my_prompt = 'Categorize the research article into one of the following two categories based on its relevance to the topic of {}. No additional explanations are necessary.\n\n###Categories:\nRelevant\nIrrelevant\n'.format(my_topic)

my_inputs = [i+". "+j for i,j in zip(CH_data["short_title"].values, CH_data["abstract"].values)]
my_inputs = [s.strip() for s in my_inputs] # remove trailing spaces
my_inputs = np.array(my_inputs)
my_outpus = CH_data["Label"].values
my_foldid = "fold_1" # fold_1 | fold_2 | fold_3
if my_foldid == "fold_0":
    print('>>> NOTE: full training -- using all data <<<')
    my_split = np.zeros(len(my_inputs))
else:
    my_split = CH_data[my_foldid].values # !! change here for K_RUN !!

#%%
idx_train = df_split(my_split,split='train')
my_dict   = {'instruction': [my_prompt]*len(idx_train),
             'input'      : copy.deepcopy(my_inputs[idx_train]),
             'output'     : my_outpus[idx_train]}
lab_dist(my_dict)

# test set
idx_test = df_split(my_split,split='test')
my_dict_ = {'instruction': [my_prompt]*len(idx_test),
             'input'      : copy.deepcopy(my_inputs[idx_test]),
             'output'     : my_outpus[idx_test]}
lab_dist(my_dict_)
#%% whole data mode
val_mode     = False # True | False
if val_mode:    
    val_data_dir = r'dili_data/camda2022/'    
    vis_mode     = False
    
    read_val = lambda fname: pd.read_csv(val_data_dir+fname, sep=',')
    
    df_test_1 = read_val('CAMDA 2022 T1 test.csv')
    df_test_2 = read_val('CAMDA 2022 T2 test.csv')
    df_test_3 = read_val('CAMDA 2022 T3 test.csv')
    
    df_val_1 = read_val('CAMDA 2022 T1 validation.csv')
    df_val_2 = read_val('CAMDA 2022 T2 validation.csv')
    df_val_3 = read_val('CAMDA 2022 T3 validation.csv')
    df_val_4 = read_val('CAMDA 2022 V4 validation.csv')
    
    assert val_mode
    def see_df(df_test_1):
        print(df_test_1.columns, '#rows:', len(df_test_1))
        return len(df_test_1)
    
    val_list = [df_test_1, df_test_2, df_test_3, 
                df_val_1, df_val_2, df_val_3, df_val_4]
    
    val_lengs = []
    for df in val_list:
        val_lengs.append(see_df(df))
    
    df_val_all = pd.concat(val_list, axis=0, ignore_index=True)
#%% !!! visualization mode !!!
vis_mode = True # False True
if vis_mode:
    ## assert my_foldid == "fold_0"
    
    val_data_dir = r'saved/visualization/' # switch between saved | saved_dili
    val_mode     = True
    
    df_val_all = pd.read_csv(val_data_dir+'dup.csv', sep=',')

#%% whole data mode, further analysis
if val_mode:
    set_names = ['test-1', 'test-2', 'test-3',
                 'val-1', 'val-2', 'val-3', 'val-4']
    sets = []
    for i in range(7):
        sets += [set_names[i]] * val_lengs[i]
    sets = np.array(sets)
    
    df_val_all['set'] = sets
#%% 
if val_mode:
    my_dict_ = {'instruction' : [my_prompt]*len(df_val_all),
                 'input'      : df_val_all['Description'].values,
                 'output'     : ['Irrelevant']*len(df_val_all)
               }

#%% !!! visualization mode !!!
if vis_mode:
    my_dict_ = {'instruction' : [my_prompt]*len(df_val_all),
                 'input'      : df_val_all['abstract'].values,
                 'output'     : ['Irrelevant']*len(df_val_all)
               }

#%% input size cut-off
parse_max_len(my_dict, 2500, replace=True)

#%%
str_hash = lambda s: hashlib.md5(s.encode('utf-8')).hexdigest()
my_hash  = np.array([str_hash(d['short_title']) for ind, d in CH_data.iloc[idx_test].iterrows()])

#%%
dataset = Dataset.from_dict(my_dict)

print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

print(dataset[randrange(len(dataset))])

# test set
if len(my_dict_['input']) == 0:
    print('>>> dummy test set, only for full data training!')
    my_dict_ = copy.deepcopy(my_dict)
dataset_ = Dataset.from_dict(my_dict_)

print(f'Number of prompts: {len(dataset_)}')
print(f'Column names are: {dataset_.column_names}')

print(dataset_[randrange(len(dataset_))])

#%%
print(create_prompt_formats(dataset[randrange(len(dataset))]))
#%% Call the preprocess_dataset function to preprocess the training dataset.
# Random seed
seed = 33

max_length = 1024 # get_max_length(model) | 512 | 1024
print('> max_length for llama-2 model:', max_length)
preprocessed_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)

print(preprocessed_dataset)

print(preprocessed_dataset[0])
#%% Initializing QLoRA and TrainingArguments parameters below for training.
#######################################################################
# QLoRA parameters
#######################################################################

# LoRA attention dimension
lora_r = 16 # 16 | 8

# Alpha parameter for LoRA scaling
lora_alpha = 64 # 64 | 32

# Dropout probability for LoRA layers
lora_dropout = 0.1

# Bias
bias = "none"

# Task type
task_type = "CAUSAL_LM"

#######################################################################
# TrainingArguments parameters
#######################################################################

# Output directory where the model predictions and checkpoints will be stored
fold_N = my_foldid[-1]
output_dir = "./results_7b_chat_k1"
print('model dir:', output_dir)

assert my_foldid[-1] == output_dir[-1] # check same fold id
print(model_name)
if 'pmc' in output_dir:
    assert 'PMC' in model_name 
    llama_name = 'pmc-llama-2-7b'
else:
    assert not ('PMC' in model_name)
    llama_name = 'llama-2-7b-chat'

# Batch size per GPU for training
per_device_train_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4 # 2e-4 | 1e-4 (13B)

# Optimizer to use
optim = "paged_adamw_32bit"

# Number of training steps (overrides num_train_epochs)
max_steps = 5 # 20000 # override
num_train_epochs = 3

# Linear warmup steps from 0 to learning_rate
warmup_steps = 200

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True

# Log every X updates steps
logging_steps = 5
#%% Calling the fine_tune function below to fine-tune or instruction-tune the pre-trained model on our preprocessed news classification instruction dataset.
if check_mode(RUN_MODE, 'train'):
    print('~~~~~ Do Training ~~~~~')
    fine_tune(model, 
              tokenizer, 
              preprocessed_dataset, 
              lora_r = lora_r, 
              lora_alpha = lora_alpha, 
              lora_dropout = lora_dropout, 
              bias = bias, 
              task_type = task_type, 
              per_device_train_batch_size = per_device_train_batch_size, 
              gradient_accumulation_steps = gradient_accumulation_steps, 
              warmup_steps = warmup_steps, 
              max_steps = max_steps, 
              num_train_epochs = num_train_epochs,
              learning_rate = learning_rate, 
              fp16 = fp16, 
              logging_steps = logging_steps, 
              output_dir = output_dir, 
              optim = optim)
    exit()
