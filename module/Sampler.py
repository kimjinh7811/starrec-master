import numpy as np
import torch

def generate_pairwise_data_from_matrix(rating_matrix, num_negatives=1, p=None):
    num_users, num_items = rating_matrix.shape
    
    users = []
    positives = []
    negatives = []
    for user in range(num_users):
        if p is None:
            start = rating_matrix.indptr[user]
            end = rating_matrix.indptr[user + 1]
            pos_index = rating_matrix.indices[start:end]
            num_positives = len(pos_index)
            if num_positives == 0:
                print('[WARNING] user %d has 0 ratings. Not generating negative samples.' % user)
                continue

            num_all_negatives = num_items - num_positives
            prob = np.full(num_items, 1 / num_all_negatives)
            prob[pos_index] = 0.0

        neg_items = np.random.choice(num_items, num_positives * num_negatives, replace=True, p=prob)
        for i, pos in enumerate(pos_index):
            users += [user] * num_negatives
            positives += [pos]  * num_negatives
            negatives += neg_items[i * num_negatives: (i + 1) * num_negatives].tolist()

    return torch.LongTensor(users), torch.LongTensor(positives), torch.LongTensor(negatives)


########################################################################################
# def generate_pairwise_data(dataset):
#     # sample negative item for all interactions.
#     # 1 neg per pos

#     user_ids, pos_ids, neg_ids = [], [], []

#     num_users = dataset.num_users
#     num_items = dataset.num_items
#     train_dict = dataset.train_dict

#     sample_p = np.ones(num_items)
#     for u in train_dict:
#         pos_items = [x[0] for x in train_dict[u]]
#         num_pos = len(pos_items)

#         # sample prob. of all items
#         sample_p_user = np.array(sample_p)
#         # pos item is not sampled
#         sample_p_user[pos_items] = 0
#         sample_p_user /= (num_items - num_pos)

#         neg_items = np.random.choice(num_items, num_pos, replace=True, p=sample_p_user)

#         user_ids += [u] * num_pos
#         pos_ids += pos_items
#         neg_ids += neg_items.tolist()
#     return torch.LongTensor(user_ids), torch.LongTensor(pos_ids), torch.LongTensor(neg_ids)
#     # return np.array(user_ids), np.array(pos_ids), np.array(neg_ids)
