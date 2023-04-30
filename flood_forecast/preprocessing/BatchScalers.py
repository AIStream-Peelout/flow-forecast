from faker import Factory
import pandas as pd
from tqdm.notebook import tqdm
import os
import numpy as np

class CustomStandardScaler():
    """
    Standard scaler for batch processing.

    Needs to be declared before the batch processing starts
    """
    """https://stats.stackexchange.com/questions/133138/will-the-mean-of-a-set-of-means-always-be-the-same-as-the-mean-obtained-from-the"""
    def __init__(self,
                 id_cols,
                 real_cols,
                 batch_size=1000):
        self.batch_size=batch_size
        self.id_cols=id_cols
        self.cols=real_cols
        self.batch_params={x:{'sum':[],'count':[]} for x in self.cols}
        self.std_params={x:{'sum':[],'count':[]} for x in self.cols}

    def split_file(path):
        return

    def get_mean_params(self,path,fname):
        """
        run this only on the training batches
        """

        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):

            sum_=batch[self.cols].sum()
            count_=batch[self.cols].count()
            for x,y,z in zip(sum_.index,sum_,count_):
                self.batch_params[x]['sum'].append(y)
                self.batch_params[x]['count'].append(z)

            del batch, sum_, count_

    def get_pop_mean(self):
        self.mean_=[]
        for x in self.cols:
            pop_sum_=np.sum(self.batch_params[x]['sum'])
            pop_count_=np.sum(self.batch_params[x]['count'])
            self.mean_.append(pop_sum_/pop_count_)

    def get_sd_params(self,path,fname):
        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            sq_diff_= (batch[self.cols]-self.mean_)**2
            sum_=sq_diff_.sum()
            count_=sq_diff_.count()
            for x,y,z in zip(sum_.index,sum_,count_):
                self.std_params[x]['sum'].append(y)
                self.std_params[x]['count'].append(z)

            del batch, sum_, count_, sq_diff_

    def get_pop_std(self):
        self.sd_=[]
        for x in self.cols:
            pop_sq_sum_=np.sum(self.std_params[x]['sum'])
            pop_n_=np.sum(self.std_params[x]['count'])
            self.sd_.append(np.sqrt(pop_sq_sum_/pop_n_))

    def fit(self,
            path,
            fname,
            save_dir,
            save_fname):

        print('computing mean...')
        self.get_mean_params(path,fname)
        self.get_pop_mean()
        print('MEAN COMPUTED')

        print('comuting std dev....')
        self.get_sd_params(path,fname)
        self.get_pop_std()
        print('STD DEV COMPUTED')

        i=0
        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            batch[self.cols]=(batch[self.cols]-self.mean_)/self.sd_
            batch[self.id_cols+self.cols].to_csv(os.path.join(save_dir,f'{save_fname}_b{i}.csv'),index=False)
            i=i+1

    def transform(self,
                  path,
                  fname,
                  save_dir,
                  save_fname):
        i=0
        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            batch[self.cols]=(batch[self.cols]-self.mean_)/self.sd_
            batch[self.cols].to_csv(os.path.join(save_dir,f'{save_fname}_b{i}.csv'),index=False)
            i=i+1
#class CustomMinMaxScaler(dict_,df)

class BatchLabelEncoder():

    def __init__(self,
                 id_cols,
                 cat_cols,
                 batch_size=1000):
        self.batch_size=batch_size
        self.id_cols=id_cols
        self.cols=cat_cols
        self.cols_dict={x:[] for x in self.cols}
        self.mapping_dict={x:[] for x in self.cols}

    def get_batch_unique(self,path,fname):
        """
        run this only on the training batches
        """

        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            for x in self.cols:
                self.cols_dict[x].extend(batch[x].astype(str).unique())#np.unique(batch[x].unique()))
            del batch

    def get_mapping(self):
        for x in self.cols:
            unique_vals_= sorted(set(self.cols_dict[x]))#np.sort(np.unique(np.concatenate(self.cols_dict[x])))
            self.mapping_dict[x]={k:v for k,v in zip(unique_vals_,range(len(unique_vals_)))}

            del unique_vals_

    def fit(self,
            path,
            fname,
            save_dir,
            save_fname):

        print('getting unique values')
        self.get_batch_unique(path,fname)
        print('generating mapping')
        self.get_mapping()

        print('fitting on data')
        i=0
        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            for x in self.cols:
                batch[x]=batch[x].astype(str).map(self.mapping_dict[x])
            batch[self.id_cols+self.cols].to_csv(os.path.join(save_dir,f'{save_fname}_b{i}.csv'),index=False)
            i=i+1

    def transform(self,
                  path,
                  fname,
                  save_dir,
                  save_fname):
        i=0
        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            for x in self.cols:
                batch[x]=batch[x].astype(str).map(self.mapping_dict[x])
            batch[self.id_cols+self.cols].to_csv(os.path.join(save_dir,f'{save_fname}_b{i}.csv'),index=False)
            i=i+1
