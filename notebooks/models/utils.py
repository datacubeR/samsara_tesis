import pandas as pd
import numpy as np

def create_val_ids(df, n= 10, id = "ts_id"):
    idx = df[id].sample(n, random_state=42) 
    return pd.concat([idx, idx.str.replace("_d","_nd")]).tolist()


def create_sequences(df, ts_id:str, vals: str, ids: str, seq_len = 10):

    uids = df[ts_id].unique()
    ts_id = df[ts_id].values
    values = df[vals].values
    ids = df[ids].values

    sequences = []
    idxs = []
    for id in uids:
        vals = values[ts_id == id]
        for i in range(len(vals)-seq_len+1):
            sequences.append(vals[i:i+seq_len])
            idxs.append(ids[i:i+seq_len])

    return sequences, idxs

# def accumulate_sequences(sequences, len_ts, n_ts, seq_len):
#     vals = np.zeros(len_ts*n_ts)

#     for i in range(n_ts):
#         for j in range(i*len_ts, i*len_ts+len_ts-seq_len):
#             vals[j:j+seq_len] += sequences[j]

#     return vals
def accumulate_sequences(sequences, len_ts=268, n_ts=200, seq_len=10):
    n_seq = len_ts-seq_len+1
    vals = np.zeros(len_ts*n_ts)

    for i in range(n_ts):
        # for j in range(i*len_ts, i*len_ts+len_ts-seq_len):
        for j in range(n_seq):
            s = (n_seq + seq_len - 1)*i + j
            e = s + seq_len
            vals[s:e] += sequences[n_seq*i+j]

    return vals

def create_divisors(len_ts, n_ts, seq_len):

    divisor = []
    for pos, i in enumerate(range(len_ts), start=1):
        if len_ts - pos < seq_len:
            val = min([len_ts - pos + 1, seq_len])
            divisor.append(val)
        elif i <= seq_len:
            val = min([pos, seq_len])
            divisor.append(val)
        else:
            divisor.append(seq_len)

    return divisor*n_ts