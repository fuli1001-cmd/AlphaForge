import pandas as pd
import torch 
from torch import nn
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# from alphagen.config import *
# from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.rl.env.core import AlphaEnvCore
import torch.nn.functional as F
from gan.dataset import Collector
from gan.network.generater import NetG_DCGAN
from gan.network.masker import NetM
from gan.network.predictor import NetP, train_regression_model,train_regression_model_with_weight
from alphagen.rl.env.wrapper import SIZE_ACTION,action2token

from alphagen_generic.features import open_
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *

from gan.utils.data import get_data_by_year


instruments: str = "csi500"
freq = 'day'
save_name = 'test'
window = float('inf')

from alphagen.utils.correlation import batch_pearsonr,batch_spearmanr
device = 'cuda:0'
result = []
pred_dfs = {}
for n_factors in [10]:
    for seed in range(5):
        cur_seed_ic = []
        cur_seed_ric = []
        all_pred_df_list = []
        for train_end in range(2020,2021):
            print(n_factors,seed,train_end)
            returned = get_data_by_year(
                train_start = 2012,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
                instruments=instruments, target=target,freq=freq,
            )
            data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
            

            path = f'out/{save_name}_{instruments}_{train_end}_{seed}/z_bld_zoo_final.pkl'
            tensor_save_path = f'out/{save_name}_{instruments}_{train_end}_{seed}/pred_{train_end}_{n_factors}_{window}_{seed}.pt'

            pred = torch.load(tensor_save_path).to(device)
            tgt = target.evaluate(data_all)
            
            
            ones = torch.ones_like(tgt)
            ones = ones * torch.nan
            ones[-data_test.n_days:] = pred
            cur_df = data_all.make_dataframe(ones)
            all_pred_df_list.append(cur_df.unstack().iloc[-data_test.n_days:].stack())
            
            tgt = tgt[-data_test.n_days:].to(device)
            
            
            ic_s = torch.nan_to_num(batch_pearsonr(pred,tgt),nan=0)
            rank_ic_s = torch.nan_to_num(batch_spearmanr(pred,tgt),nan=0)

            print(f"**** data_test.n_days: {data_test.n_days}, pred shape: {pred.shape}, tgt.shape: {tgt.shape}, ic_s: {ic_s.shape}, rank_ic_s: {rank_ic_s.shape}")
            print(data_test._dates[data_test.max_backtrack_days : -data_test.max_future_days])

            cur_seed_ic.append(ic_s)
            cur_seed_ric.append(rank_ic_s)
            
        pred_dfs[f"{n_factors}_{seed}"] = pd.concat(all_pred_df_list,axis=0)
        ic = torch.cat(cur_seed_ic)
        rank_ic = torch.cat(cur_seed_ric)

        ic_mean = ic.mean().item()
        rank_ic_mean = rank_ic.mean().item()
        ic_std = ic.std().item()
        rank_ic_std = rank_ic.std().item()
        print(f"ic shape: {ic.shape}, rank ic shape: {rank_ic.shape}")
        if ic_std == 0:
            print(f"---- IC std is zero for seed {seed}, num {n_factors}")
        if rank_ic_std == 0:
            print(f"---- Rank IC std is zero for seed {seed}, num {n_factors}")
        tmp = dict(
            seed = seed,
            num = n_factors,
            ic = ic_mean,
            ric = rank_ic_mean,
            icir = ic_mean/ic_std if ic_std != 0 else 0.,
            ricir = rank_ic_mean/rank_ic_std if rank_ic_std != 0 else 0.,
        )
        result.append(tmp)

run_result = pd.DataFrame(result).groupby(['num','seed']).mean().groupby('num').agg(['mean','std'])
print(run_result)