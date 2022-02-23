### DeSurv example: train a DeSurv model on the SUPPORT dataset ###

import numpy as np
import pandas as pd
import torch

from pycox.datasets import support
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from torch.utils.data import TensorDataset, DataLoader

from classes import ODESurvSingle


df_train = support.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
cols_leave = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
y_test = get_target(df_test)

t_train, e_train = y_train
t_val, e_val = y_val
t_test, e_test = y_test

t_train_max = np.amax(t_train)
t_train = t_train / t_train_max
t_val = t_val / t_train_max
t_test = t_test / t_train_max

batch_size = 32
dataset_train = TensorDataset(*[torch.tensor(u,dtype=dtype_) for u, dtype_ in [(x_train,torch.float32),
                                                                               (t_train,torch.float32),
                                                                               (e_train,torch.long)]])
data_loader_train = DataLoader(dataset_train, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)

dataset_val = TensorDataset(*[torch.tensor(u,dtype=dtype_) for u, dtype_ in [(x_val,torch.float32),
                                                                               (t_val,torch.float32),
                                                                               (e_val,torch.long)]])
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, shuffle=True)

dataset_test = TensorDataset(*[torch.tensor(u,dtype=dtype_) for u, dtype_ in [(x_test,torch.float32),
                                                                               (t_test,torch.float32),
                                                                               (e_test,torch.long)]])
data_loader_test = DataLoader(dataset_test, batch_size=batch_size, pin_memory=True, shuffle=True)


hidden_dim = 32
training = False

lr = 1e-3

xdim = x_train.shape[1]

model = ODESurvSingle(lr, xdim, hidden_dim)

if training:
    model.optimize(data_loader_train, n_epochs=300, logging_freq=1, data_loader_val=data_loader_val, max_wait=20)
    torch.save(model.state_dict(), "tst_model")
    model.eval()
else:
    state_dict = torch.load("tst_model")
    model.load_state_dict(state_dict)
    model.eval()

argsortttest = np.argsort(t_test)
t_test = t_test[argsortttest]
e_test = e_test[argsortttest]
x_test = x_test[argsortttest,:]

n_eval = 3000
t_eval = np.linspace(0,np.amax(t_test),n_eval)

with torch.no_grad():
    t_ = torch.tensor(np.concatenate([t_eval]*x_test.shape[0], 0),dtype=torch.float32)
    x_ = torch.tensor(np.repeat(x_test, [t_eval.size]*x_test.shape[0], axis=0), dtype=torch.float32)
    surv = pd.DataFrame(np.transpose((1 - model.predict(x_,t_).reshape((x_test.shape[0],t_eval.size))).detach().numpy()),
                        index=t_eval)

ev = EvalSurv(surv, t_test, e_test, censor_surv='km')

time_grid = np.linspace(t_test.min(), 0.9*t_test.max(), 1000)
print(ev.concordance_td())
print(ev.integrated_brier_score(time_grid))
print(ev.integrated_nbll(time_grid))
