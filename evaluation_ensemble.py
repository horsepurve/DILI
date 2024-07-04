#%% ========== ========== ========== ========== ========== ==========
#
# 2nd stage ML
# 

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from text_modeling_help import chk_metr

#%%
save_dir = 'saved_dili'
dataset  = 'DILI' 
RAN_SEED = 2333

#%% load finel
file_name = "{}/{}_{}.csv".format(save_dir, 
                                  dataset,
                                  RAN_SEED)      
print('--> loading final:', file_name)
df_all = pd.read_csv(file_name)

print(df_all.columns)
print(len(df_all.columns)) # 39

#%% check the out-of-fold metrics
y_test = np.array([1 if i.startswith('R') else 0 for i in df_all['Label']])

chk_metr(y_test, df_all['specter'].values)
chk_metr(y_test, df_all['PubMedBERT'].values)
chk_metr(y_test, df_all['BioGPT'].values)
chk_metr(y_test, df_all['pmc-llama-2-7b'].values)
chk_metr(y_test, df_all['llama-2-7b-chat'].values)
# chk_metr(y_test, df_all['pmc-llama-2-7b-b'].values) # 0-1 prediction
# chk_metr(y_test, df_all['llama-2-7b-chat-b'].values) # 0-1 prediction

chk_metr(y_test, df_all['BERT'].values)
chk_metr(y_test, df_all['ALBERT'].values)
chk_metr(y_test, df_all['DistilBERT'].values)
chk_metr(y_test, df_all['RoBERTa'].values)
chk_metr(y_test, df_all['GPT'].values)
chk_metr(y_test, df_all['GPT2'].values)
chk_metr(y_test, df_all['Bio_ClinicalBERT'].values)
#%% prepare the features
feats = ['leng', \
         'specter', 'PubMedBERT', 'BioGPT', \
         'pmc-llama-2-7b', 'llama-2-7b-chat', \
         'pmc-llama-2-7b-b', 'llama-2-7b-chat-b', \
         'BERT', 'ALBERT', 'DistilBERT', 'RoBERTa', 'GPT', 'GPT2', \
         'Bio_ClinicalBERT', \
         'n_PubMedBERT', 'n_Bio_ClinicalBERT', 'n_BERT', 'n_ALBERT', \
         'n_DistilBERT', 'n_RoBERTa', 'n_GPT', 'n_GPT2', 'n_BioGPT', \
         'n_Llama-2-7b-chat-hf', \
         ] # 'has_CL', 'has_topic'
X_all = df_all[feats].values
y_all = y_test

# num_tokens / len_doc
for col in [15,16,17,18,19,20,21,22,23,24]:    
    X_all[:,col] = X_all[:,col] / X_all[:,0]
#%%
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
from scipy.special import expit

#%%
# ~~~ full K-fold pipeline see below ~~~

#%% train/test split

train_X, test_X, train_y, test_y = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42)

print('train shapes:', train_X.shape, train_y.shape, 
      'test shapes:', test_X.shape, test_y.shape)

#%% prepare final score table
scores_all = {'specter': list(chk_metr(test_y, test_X[:,1])),
              'PubMedBERT': list(chk_metr(test_y, test_X[:,2])),
              'BioGPT': list(chk_metr(test_y, test_X[:,3])),
              'pmc-llama-2-7b': list(chk_metr(test_y, test_X[:,4])),
              'llama-2-7b-chat': list(chk_metr(test_y, test_X[:,5]))         
              }

scores_all = pd.DataFrame.from_dict(scores_all, orient='index',
  columns=['AUC', 'ACC', 'TP', 'FP', 'FN', 'TN', 'Recall', 'Precision'])
#%% simple ensemble
ens_4 = (test_X[:,2] + test_X[:,3] + test_X[:,4] + test_X[:,5]) * 0.25

scores_all.loc['ensemble(4)'] = list(chk_metr(test_y, ens_4))

#%% Logistic Regression
n_fold   = 5
kf       = KFold(n_splits = n_fold, shuffle = True, random_state = 42)
pred     = 0
results  = np.zeros((train_X.shape[0],))
mean_acc = 0

log_reg = LogisticRegression(n_jobs = -1, C = 0.01, penalty = 'l2', random_state = 42)

for fold, (train_id, valid_id) in enumerate(kf.split(train_X)):
    X_train, X_val = train_X[train_id], train_X[valid_id]
    y_train, y_val = train_y[train_id], train_y[valid_id]
    
    log_reg.fit(X_train, y_train,
             )
    
    # Out of Fold predictions
    results=  log_reg.predict_proba(X_val) 
    
    pred += log_reg.predict_proba(test_X)[:,1] / n_fold
    
    fold_acc = roc_auc_score(y_val ,results[:,1])
    
    print(f"Fold {fold} | Fold accuracy: {fold_acc}")
    
    mean_acc += fold_acc / n_fold
    
print(f"\nOverall AUC: {mean_acc}")

chk_metr(test_y, pred)
#%% note that this is NOT the out-of-fold score!
scores_all.loc['LR'] = list(chk_metr(test_y, pred))

#%%
xgb_params = {
    'max_depth': 4,
    'booster': 'gbtree', 
    'n_estimators': 10000,
    'random_state': 42,
    'tree_method':'hist',
    'device': 'cuda',
    'eval_metric': "error", # error auc
    'predictor': "gpu_predictor",
    'early_stopping_rounds': 100
}

#%% https://www.kaggle.com/code/ninjaac/xgboost-vs-logisticregression-optuna-features

n_fold   = 15
kf       = KFold(n_splits = n_fold, shuffle = True, random_state = 42)
pred     = 0
results  = np.zeros((train_X.shape[0],))
mean_acc = 0

xgb_model = XGBClassifier(**xgb_params)

for fold, (train_id, valid_id) in enumerate(kf.split(train_X)):
    X_train, X_val = train_X[train_id],train_X[valid_id]
    y_train, y_val = train_y[train_id], train_y[valid_id]
    
    xgb_model.fit(X_train, y_train,
             verbose = False,
             eval_set = [(X_train, y_train), (X_val, y_val)]
             )
    
    # Out of Fold predictions
    results = xgb_model.predict_proba(X_val) 
    
    pred += xgb_model.predict_proba(test_X)[:,1] / n_fold
    
    fold_acc = roc_auc_score(y_val ,results[:,1])
    
    print(f"Fold {fold} | Fold accuracy: {fold_acc}")
    
    mean_acc += fold_acc / n_fold
    
print(f"\nOverall AUC: {mean_acc}")

chk_metr(test_y, pred)
#%%
scores_all.loc['XGB'] = list(chk_metr(test_y, pred))

#%%
sns.heatmap(scores_all[['AUC', 'ACC', 'Recall', 'Precision']], 
            annot=True, cmap='viridis', fmt=".2%")
