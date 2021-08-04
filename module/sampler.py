import numpy as np

def generate_pairwise_data(dataset):
    # sample negative item for all interactions.
    # 1 neg per pos

    user_ids, pos_ids, neg_ids = [], [], []

    num_users = dataset.num_users
    num_items = dataset.num_items
    train_dict = dataset.train_dict

    sample_p = np.ones(num_items)
    for u in train_dict:
        pos_items = [x[0] for x in train_dict[u]]
        num_pos = len(pos_items)

        # sample prob. of all items
        sample_p_user = np.array(sample_p)
        # pos item is not sampled
        sample_p_user[pos_items] = 0
        sample_p_user /= (num_items - num_pos)

        neg_items = np.random.choice(num_items, num_pos, replace=True, p=sample_p_user)

        user_ids += [u] * num_pos
        pos_ids += pos_items
        neg_ids += neg_items.tolist()

    return np.array(user_ids), np.array(pos_ids), np.array(neg_ids)

def generate_rating_matrix(dataset):
    pass

def generate_pointwise_data(dataset):
    pass
