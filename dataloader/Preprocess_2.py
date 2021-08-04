import os
import math
import pickle

import numpy as np
import scipy.sparse as sp
import copy

def read_raw_UIRT(datapath, separator, order_by_popularity=True):
    """
    read raw data (ex. ml-100k.rating)

    return U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, num_users, num_items, num_ratings

    """
    
    print("Loading the dataset from \"%s\"" % datapath)
    with open(datapath, "r") as f:
        lines = f.readlines()

    num_users, num_items = 0, 0
    user_to_num_items, item_to_num_users = {}, {}
    user_id_dict, item_id_dict = {}, {}

    for line in lines:
        user_id, item_id, _, _ = line.strip().split(separator)
        user_id = int(user_id)
        item_id = int(item_id)

        if user_id not in user_id_dict:
            user_id_dict[user_id] = num_users
            new_user_id = user_id_dict[user_id]

            user_to_num_items[new_user_id] = 1
            num_users += 1
        else:
            new_user_id = user_id_dict[user_id]
            user_to_num_items[new_user_id] += 1

        # Update the number of ratings per item
        if item_id not in item_id_dict:
            item_id_dict[item_id] = num_items
            
            new_item_id = item_id_dict[item_id]
            item_to_num_users[new_item_id] = 1
            num_items += 1
        else:
            new_item_id = item_id_dict[item_id]
            item_to_num_users[new_item_id] += 1

    if order_by_popularity:
        user_id_dict, user_to_num_items = order_id_by_popularity(user_id_dict, user_to_num_items)
        item_id_dict, item_to_num_users = order_id_by_popularity(item_id_dict, item_to_num_users)
    
    # BUILD U2IRTs
    U2IRT = {u: [] for u in user_to_num_items}
    for line in lines:
        user_id, item_id, rating, time = line.strip().split(separator)
        user_id = int(user_id)
        item_id = int(item_id)
        rating = float(rating)
        time = int(time)
        U2IRT[user_id_dict[user_id]].append((item_id_dict[item_id], rating, time))
    
    return U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users

def order_id_by_popularity(object_id_dict, object_to_num):
    old_to_pop_dict = {}
    new_to_pop_dict = {}
    new_object_to_num = {}
    object_id_dict_sorted = sorted(object_to_num.items(), key=lambda x: x[-1], reverse=True)
    for pop, new_pop_tuple in enumerate(object_id_dict_sorted):
        new = new_pop_tuple[0]
        new_to_pop_dict[new] = pop
        new_object_to_num[pop] = object_to_num[new]
    for old, new in object_id_dict.items():
        old_to_pop_dict[old] = new_to_pop_dict[new]

    return old_to_pop_dict, new_object_to_num

def filter_min_item_cnt(U2IRT, min_item_cnt, user_id_dict):
    modifier=0

    user_id_list = []
    for id in user_id_dict:
        user_id_list.append(id)

    for old_user_id in user_id_list:
        new_user_id = user_id_dict[old_user_id]
        IRTs = U2IRT[new_user_id]
        num_items = len(IRTs)

        if num_items < min_item_cnt:
            U2IRT.pop(new_user_id)
            user_id_dict.pop(old_user_id)
            modifier += 1
        elif modifier > 0:
            U2IRT[new_user_id - modifier] = IRTs
            user_id_dict[old_user_id] = new_user_id - modifier
    return U2IRT, user_id_dict
    
def preprocess(raw_file, file_prefix, split_type, train_ratio, valid_ratio, split_random, min_item_cnt, separator, order_by_popularity=True):
    """
    read raw data and preprocess

    """

    # read raw data
    U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users = read_raw_UIRT(raw_file, separator, order_by_popularity)

    # filter out (min item cnt)
    if min_item_cnt > 0:
        U2IRT, user_id_dict = filter_min_item_cnt(U2IRT, min_item_cnt, user_id_dict)

    # preprocess
    if split_type == 'loo':
        preprocess_loo(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, file_prefix)
    elif split_type == 'holdout':
        preprocess_holdout(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, train_ratio, valid_ratio, split_random, file_prefix)
    elif split_type == 'user':
        preprocess_user(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, train_ratio, valid_ratio, file_prefix)
    else:
        raise ValueError("Incorrect value has passed for split type: %s" % split_type)

def preprocess_loo(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users,file_prefix):
    """
    preprocess, split and save

    return None

    """
    # preprocess
    train_data = {u: [] for u in U2IRT}
    valid_data = {u: [] for u in U2IRT}
    test_data = {u: [] for u in U2IRT}
    
    num_users = len(user_id_dict)
    num_items = len(item_id_dict)
    num_ratings = 0

    for user in U2IRT:
        # Sort the UIRTs by the ascending order of the timestamp.
        IRTs = sorted(U2IRT[user], key=lambda x: x[-1])
        num_ratings += len(IRTs)

        # test data
        test_IRT = IRTs.pop()
        test_data[user] = [test_IRT]

        # valid data
        valid_IRT = IRTs.pop()
        valid_data[user] = [valid_IRT]

        # train data
        train_data[user] = IRTs
    
    # save
    data_to_save = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data,
        'num_users': num_users,
        'num_items': num_items
    }

    info_to_save = {
        'user_id_dict': user_id_dict,
        'user_to_num_items': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_to_num_users': item_to_num_users
    }

    ratings_per_user = [len(U2IRT[u]) for u in U2IRT]

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f" % (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))
    
    with open(file_prefix + '.stat', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    with open(file_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess leave-one-out finished.')


def preprocess_holdout(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, train_ratio, valid_ratio, split_random, file_prefix):
    train_data = {u: [] for u in U2IRT}
    valid_data = {u: [] for u in U2IRT}
    test_data = {u: [] for u in U2IRT}
    
    num_users = len(user_id_dict)
    num_items = len(item_id_dict)
    num_ratings = 0

    for user in U2IRT:
        # Sort the UIRTs by the ascending order of the timestamp.
        IRTs = sorted(U2IRT[user], key=lambda x: x[-1])
        num_ratings_cur_user = len(IRTs)
        num_ratings += num_ratings_cur_user
        

        train_num = int(num_ratings_cur_user * train_ratio)
        test_num = num_ratings_cur_user - train_num
        valid_num = int(train_num * valid_ratio)
        train_num -= valid_num

        assert num_ratings_cur_user == (train_num + valid_num + test_num)

        if split_random:
            perm = np.random.permutation(num_ratings_cur_user)
        else:
            perm = np.arange(num_ratings_cur_user)
        
        # train data
        train_idx = perm[:train_num]
        train_data[user] = [IRTs[i] for i in train_idx]

        # valid data
        valid_idx = perm[train_num: train_num + valid_num]
        valid_data[user] = [IRTs[i] for i in valid_idx]
        
        # test data
        test_idx = perm[train_num + valid_num:]
        test_data[user] = [IRTs[i] for i in test_idx]
    
    # save
    data_to_save = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data,
        'num_users': num_users,
        'num_items': num_items
    }

    info_to_save = {
        'user_id_dict': user_id_dict,
        'user_to_num_items': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_to_num_users': item_to_num_users
    }

    ratings_per_user = [len(U2IRT[u]) for u in U2IRT]

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f" % (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))
    
    with open(file_prefix + '.stat', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    with open(file_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess holdout finished.')


def preprocess_user(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, train_ratio, valid_ratio, file_prefix):

    num_users = len(user_id_dict)
    num_items = len(item_id_dict)
    num_ratings = 0

    # shuffle the user idx
    perm_user_idx = np.random.permutation(num_users)

    train_user_idx = perm_user_idx[:int(num_users * train_ratio)]
    valid_user_idx = perm_user_idx[int(num_users * train_ratio):int(num_users * (train_ratio + valid_ratio))]
    test_user_idx = perm_user_idx[int(num_users * (train_ratio + valid_ratio)): ]

    assert len(train_user_idx) + len(valid_user_idx) + len(test_user_idx) == num_users

    train_data = {u: [] for u in train_user_idx}
    valid_input_data = {u: [] for u in valid_user_idx}
    valid_target_data = {u: [] for u in valid_user_idx}
    test_input_data = {u: [] for u in test_user_idx}
    test_target_data = {u: [] for u in test_user_idx}

    for user in U2IRT:
        # Sort the UIRTs by the ascending order of the timestamp.
        IRTs = sorted(U2IRT[user], key=lambda x: x[-1])
        num_ratings_cur_user = len(IRTs)
        num_ratings += num_ratings_cur_user

        if user in train_user_idx:
            train_data[user] = IRTs

        elif user in valid_user_idx:
            target_IRT = IRTs.pop()
            valid_target_data[user] = target_IRT

            valid_input_data[user] = IRTs

        elif user in test_user_idx:
            target_IRT = IRTs.pop()
            test_target_data[user] = target_IRT

            test_input_data[user] = IRTs
    # save
    data_to_save = {
        'train': train_data,
        'valid_input': valid_input_data,
        'valid_target': valid_target_data,
        'test_input': test_input_data,
        'test_target': test_target_data,
        'num_users': num_users,
        'num_items': num_items
    }

    info_to_save = {
        'user_id_dict': user_id_dict,
        'user_to_num_items': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_to_num_users': item_to_num_users
    }

    ratings_per_user = [len(U2IRT[u]) for u in U2IRT]

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f" % (
    min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))

    with open(file_prefix + '.stat', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)

    with open(file_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess user-partition finished.')