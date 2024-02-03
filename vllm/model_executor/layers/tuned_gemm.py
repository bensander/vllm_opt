import torch
import torch.nn.functional as F
from rocsolidxgemm import rocb_create_extension,rocb_mm
from hipbsolidxgemm import hipb_create_extension,hipb_mm
from pathlib import Path
import os
import yaml
import pandas as pd
from vllm import custom_ops


class TunedGemm:
    def __init__(self):
        #rocb_create_extension()
        #hipb_create_extension()
        self.extensions_created = False
        self.bestsols = {}
        self.load_best_sols()
        self.create_ds()
        self.save_gemm = int(os.environ.get('VLLM_TUNE_GEMM',0))
        self.untune_path = os.environ.get('VLLM_UNTUNE_FILE', "/tmp/vllm_untuned.csv")

        if (self.save_gemm == 1):
            if (Path(self.untune_path).is_file()):
                self.tuned_df = pd.read_csv(self.untune_path)
            else:
                self.tuned_df = pd.DataFrame(columns=['M','N','K'])
        else:
            self.tuned_df = None

    def load_best_sols(self):
        perfbits = {}
        perf_file = os.environ.get('VLLM_PERF_YAML')
        if perf_file is not None:
            with open(perf_file, 'r') as file:
                perfbits = yaml.safe_load(file)

        tune_file = perfbits.get('tuned_gemm_csv',None)
        if tune_file is not None and Path(tune_file).is_file():
            self.bestsols = pd.read_csv(tune_file)


    def apply_custom(self,ds):
        M,N,K = ds['M'],ds['N'],ds['K']
        #apply custom matvec (only for f16 dtype)
        if N==1:
            ds1 = ds.copy()
            ds1['libtype'] = 'custom'
            if K==8192 and (M==1280 or M==7168):
                ds1['solidx'] = 8
                return ds1
            elif K==3584 and M==8192:
                ds1['solidx'] = 8
                return ds1
            elif K<=8192 and K%8==0 and M%4==0:
                ds1['solidx'] = 1
                return ds1
        return ds
    def create_ds(self):
        df = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = self.apply_custom(df.iloc[i])
            key = (ds['M'],ds['N'],ds['K'])
            if ds['libtype']=='hipblaslt': soltype = 1
            elif ds['libtype']=='rocblas': soltype = 2
            elif ds['libtype']=='custom': soltype = 3
            solds[key] = (soltype,int(ds['solidx']))
        self.solids = solds
        #print('>>>',solds)
    def query_sol(self,m,n,k):
        return self.solids.get((m,n,k),(0,0))
    def mm(self,inp,weights):
        if self.extensions_created == False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
        soltype,solidx = self.query_sol(m=weights.shape[0],n=inp.shape[0],k=inp.shape[1])
        if soltype==1:
            out = hipb_mm(inp,weights.t(),solidx)
        elif soltype==3:
            ##only matvec is supported currently
            out = torch.empty(inp.shape[0],weights.shape[0],dtype=torch.float16,device='cuda')
            #print('>>>Matvec',inp.shape,weights.shape,soltype,solidx)
            if solidx<=1:
                custom_ops.LLMM1(weights,inp,out,4)
            elif solidx==2:
                custom_ops.LLMM1(weights,inp,out,2)
            elif solidx==8:
                custom_ops.LLMM1(weights,inp,out,8)
            elif solidx==20:
                custom_ops.LLZZ(weights,inp,out,0)
            elif solidx==21:
                custom_ops.LLZZ(weights,inp,out,1)
        elif soltype==2:
            out = rocb_mm(inp,weights.t(),solidx)
        else:
            #print('>>>Tgemm Default',inp.shape,weights.shape,soltype,solidx)
            if (self.save_gemm == 1):
                self.tuned_df = pd.concat([self.tuned_df, pd.DataFrame({'M':[weights.shape[0]], 'N':[inp.shape[0]], 'K':[inp.shape[1]]})]).drop_duplicates()
                self.tuned_df.to_csv(self.tune_path, index=False)
            out = F.linear(inp,weights)
        return out

tgemm = TunedGemm()