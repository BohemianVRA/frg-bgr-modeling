import numpy as np

def split_train_val_instances(ids, split):
    idx_full = np.array(ids)

    np.random.seed(0)
    np.random.shuffle(idx_full)


    len_valid = int(len(ids) * split)

    valid_idx = idx_full[0:len_valid].tolist()
    train_idx = np.delete(idx_full, np.arange(0, len_valid)).tolist()

    return train_idx, valid_idx
