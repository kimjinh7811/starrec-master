import os
import math
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
   
def preprocess(raw_file, file_prefix, split_type, split_random, test_ratio, valid_ratio, holdout_users, leave_k, implicit=True, binarize_threshold=1.0, min_item_per_user=0, min_user_per_item=0, separator=',', order_by_popularity=True):
    """
    read raw data and preprocess

    """
    print("Loading the dataset from \"%s\"" % raw_file)
    # data: pandas dataframe
    data = pd.read_csv(raw_file, sep=separator, names=['user', 'item', 'ratings', 'timestamps'],
                                                dtype={'user': int, 'item': int,  'ratings': float, 'timestamps': float},
                                                engine='python')

    

    # initial # user, items
    num_users = len(pd.unique(data.user))
    num_items = len(pd.unique(data.item))

    print('initial user, item:', num_users, num_items)
    
    if implicit:
        print("Binarize ratings greater than or equal to %.f" % binarize_threshold)
        data = data[data['ratings'] >= binarize_threshold]
        data['ratings'] = 1.0




     # initial # user, items
    num_users = len(pd.unique(data.user))
    num_items = len(pd.unique(data.item))

    print('initial user, item:', num_users, num_items)



    # filter users
    num_items_by_user = data.groupby('user', as_index=False).size()
    user_filter_idx = data['user'].isin(num_items_by_user[num_items_by_user['size'] >= min_item_per_user]['user'])
    data = data[user_filter_idx]
    num_items_by_user = data.groupby('user', as_index=False).size()

    num_users = len(pd.unique(data.user))
    print('# user after filter (min %d items): %d' % (min_item_per_user, num_users))



    # filter items
    num_users_by_item = data.groupby('item', as_index=False).size()
    item_filter_idx = data['item'].isin(num_users_by_item[num_users_by_item['size'] >= min_user_per_item]['item'])
    data = data[item_filter_idx]
    num_users_by_item = data.groupby('item', as_index=False).size()

    num_items = len(pd.unique(data.item))
    print('# item after filter (min %d users): %d' % (min_user_per_item, num_items))



    # assign new user id
    print('Assign new user id...')
    user_frame = num_items_by_user
    if order_by_popularity: 
        user_frame = user_frame.sort_values(by='size', ascending=False)
    user_frame['new_id'] = list(range(num_users))

    frame_dict = user_frame.to_dict()
    user_id_dict = {k:v for k, v in zip(user_frame['user'], user_frame['new_id'])}
    user_frame = user_frame.set_index('new_id')
    user_to_num_items = user_frame.to_dict()['size']
    data.user = [user_id_dict[x] for x in  data.user.tolist()]



    # assign new item id
    print('Assign new item id...')
    item_frame = num_users_by_item
    if order_by_popularity: 
        item_frame = item_frame.sort_values(by='size', ascending=False)
    item_frame['new_id'] = list(range(num_items))


    frame_dict = item_frame.to_dict()
    item_id_dict = {k:v for k, v in zip(item_frame['item'], item_frame['new_id'])}
    item_to_num_users = item_frame.to_dict()['size']
    data.item = [item_id_dict[x] for x in  data.item.tolist()]




    # preprocess
    print('Split data into train, val, test... (%s)' % split_type)
    if split_type == 'hold-user-out':
        preprocess_hold_user_out(data, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, holdout_users, test_ratio, valid_ratio, split_random, file_prefix)
    elif split_type == 'hold-out-user-5-fold-test': # jhkim, jwlee
        preprocess_hold_user_out_5_fold_test(data, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, holdout_users, test_ratio, valid_ratio, split_random, file_prefix)
    elif split_type in ['holdout', 'loo', 'lko']:
        preprocess_others(data, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, test_ratio, valid_ratio, leave_k, split_type, split_random, file_prefix)
    else:
        raise ValueError("Incorrect value has passed for split type: %s" % split_type)
    
    print('Preprocess finished...!')

def preprocess_hold_user_out_5_fold_test(data, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, holdout_users=100, test_ratio=0.2, valid_ratio=0.1, split_random=True, file_prefix='data'): # jhkim, jwlee
    num_users = len(user_id_dict)
    num_items = len(item_id_dict)
    num_ratings = len(data)

    if isinstance(holdout_users, float):
        holdout_users = int(num_users * holdout_users)

    num_train_users = num_users - holdout_users * 2
    assert num_users == num_train_users + holdout_users * 2

    user_perm = np.random.permutation(num_users)
    train_users = user_perm[:num_train_users]
    valid_users = user_perm[num_train_users: num_train_users + holdout_users]
    test_users = user_perm[num_train_users + holdout_users:]

    #train_data = data.loc[data.user.isin(train_users)]
    #valid_data = data.loc[data.user.isin(valid_users)]
    #test_data = data.loc[data.user.isin(test_users)]
    
    # jw code
    
    train_data = data.loc[data.user.isin(train_users)]
    uni_item = pd.unique(train_data.item)

    valid_data = data.loc[data.user.isin(valid_users)]
    valid_data = valid_data.loc[valid_data.item.isin(uni_item)]

    test_data = data.loc[data.user.isin(test_users)]
    test_data = test_data.loc[test_data.item.isin(uni_item)]


    valid_input, valid_target = split_train_test(valid_data, test_ratio=test_ratio, split_random=split_random)
    test_5_fold_input, test_5_fold_target = split_train_test_5_fold(test_data, split_random=split_random) #########

    # train_dict = df_to_dict(train_data)
    # valid_input_dict = df_to_dict(valid_input)
    # valid_target_dict = df_to_dict(valid_target)
    # test_input_dict = df_to_dict(test_input)
    # test_target_dict = df_to_dict(test_target)

    train_sp_matrix = df_to_sparse(train_data, shape=(num_users, num_items))
    valid_input_sp_matrix = df_to_sparse(valid_input, shape=(num_users, num_items))
    valid_target_sp_matrix = df_to_sparse(valid_target, shape=(num_users, num_items))

    test_input_sp_matrix_5_fold = []
    test_target_sp_matrix_5_fold = []
    for i in range(5):
        test_input_sp_matrix_5_fold.append(df_to_sparse(test_5_fold_input[i], shape=(num_users, num_items)))
        test_target_sp_matrix_5_fold.append(df_to_sparse(test_5_fold_target[i], shape=(num_users, num_items)))

    ratings_per_user = list(user_to_num_items.values())

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f\n" % (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))

    info_lines.append('# train users: %d, # train ratings: %d' % (train_sp_matrix.shape[0], train_sp_matrix.nnz))
    info_lines.append('# valid_input users: %d, # valid_input ratings: %d' % (valid_input_sp_matrix.shape[0], valid_input_sp_matrix.nnz))
    info_lines.append('# valid_target users: %d, # valid_target ratings: %d' % (valid_target_sp_matrix.shape[0], valid_target_sp_matrix.nnz))
    info_lines.append('# 1-fold test_input users: %d, # test_input ratings: %d' % (test_input_sp_matrix_5_fold[0].shape[0], test_input_sp_matrix_5_fold[0].nnz)) #########
    info_lines.append('# 1-fold test_target users: %d, # test_target ratings: %d' % (test_target_sp_matrix_5_fold[0].shape[0], test_target_sp_matrix_5_fold[0].nnz)) #########
    info_lines.append('# 2-fold test_input users: %d, # test_input ratings: %d' % (test_input_sp_matrix_5_fold[1].shape[0], test_input_sp_matrix_5_fold[1].nnz)) #########
    info_lines.append('# 2-fold test_target users: %d, # test_target ratings: %d' % (test_target_sp_matrix_5_fold[1].shape[0], test_target_sp_matrix_5_fold[1].nnz)) #########
    info_lines.append('# 3-fold test_input users: %d, # test_input ratings: %d' % (test_input_sp_matrix_5_fold[2].shape[0], test_input_sp_matrix_5_fold[2].nnz)) #########
    info_lines.append('# 3-fold test_target users: %d, # test_target ratings: %d' % (test_target_sp_matrix_5_fold[2].shape[0], test_target_sp_matrix_5_fold[2].nnz)) #########
    info_lines.append('# 4-fold test_input users: %d, # test_input ratings: %d' % (test_input_sp_matrix_5_fold[3].shape[0], test_input_sp_matrix_5_fold[3].nnz)) #########
    info_lines.append('# 4-fold test_target users: %d, # test_target ratings: %d' % (test_target_sp_matrix_5_fold[3].shape[0], test_target_sp_matrix_5_fold[3].nnz)) #########
    info_lines.append('# 5-fold test_input users: %d, # test_input ratings: %d' % (test_input_sp_matrix_5_fold[4].shape[0], test_input_sp_matrix_5_fold[4].nnz)) #########
    info_lines.append('# 5-fold test_target users: %d, # test_target ratings: %d' % (test_target_sp_matrix_5_fold[4].shape[0], test_target_sp_matrix_5_fold[4].nnz)) #########

    data_to_save = {
        'train': train_sp_matrix,
        'valid_input': valid_input_sp_matrix,
        'valid_target': valid_target_sp_matrix,
        'test_input': test_input_sp_matrix_5_fold, #########
        'test_target': test_target_sp_matrix_5_fold, #########
        'num_users': num_users,
        'num_items': num_items
    }

    info_to_save = {
        'user_id_dict': user_id_dict,
        'user_to_num_items': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_to_num_users': item_to_num_users
    }

    with open(file_prefix + '.stat', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    with open(file_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess hold_user_out finished.')






def preprocess_hold_user_out(data, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, holdout_users=100, test_ratio=0.2, valid_ratio=0.1, split_random=True, file_prefix='data'):
    num_users = len(user_id_dict)
    num_items = len(item_id_dict)
    num_ratings = len(data)

    if isinstance(holdout_users, float):
        holdout_users = int(num_users * holdout_users)

    num_train_users = num_users - holdout_users * 2
    assert num_users == num_train_users + holdout_users * 2

    user_perm = np.random.permutation(num_users)
    train_users = user_perm[:num_train_users]
    valid_users = user_perm[num_train_users: num_train_users + holdout_users]
    test_users = user_perm[num_train_users + holdout_users:]

    train_data = data.loc[data.user.isin(train_users)]
    valid_data = data.loc[data.user.isin(valid_users)]
    test_data = data.loc[data.user.isin(test_users)]

    valid_input, valid_target = split_train_test(valid_data, test_ratio=test_ratio, split_random=split_random)
    test_input, test_target = split_train_test(test_data, test_ratio=test_ratio, split_random=split_random)

    # train_dict = df_to_dict(train_data)
    # valid_input_dict = df_to_dict(valid_input)
    # valid_target_dict = df_to_dict(valid_target)
    # test_input_dict = df_to_dict(test_input)
    # test_target_dict = df_to_dict(test_target)

    train_sp_matrix = df_to_sparse(train_data, shape=(num_users, num_items))
    valid_input_sp_matrix = df_to_sparse(valid_input, shape=(num_users, num_items))
    valid_target_sp_matrix = df_to_sparse(valid_target, shape=(num_users, num_items))
    test_input_sp_matrix = df_to_sparse(test_input, shape=(num_users, num_items))
    test_target_sp_matrix = df_to_sparse(test_target, shape=(num_users, num_items))

    ratings_per_user = list(user_to_num_items.values())

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f\n" % (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))

    info_lines.append('# train users: %d, # train ratings: %d' % (train_sp_matrix.shape[0], train_sp_matrix.nnz))
    info_lines.append('# valid_input users: %d, # valid_input ratings: %d' % (valid_input_sp_matrix.shape[0], valid_input_sp_matrix.nnz))
    info_lines.append('# valid_target users: %d, # valid_target ratings: %d' % (valid_target_sp_matrix.shape[0], valid_target_sp_matrix.nnz))
    info_lines.append('# test_input users: %d, # test_input ratings: %d' % (test_input_sp_matrix.shape[0], test_input_sp_matrix.nnz))
    info_lines.append('# test_target users: %d, # test_target ratings: %d' % (test_target_sp_matrix.shape[0], test_target_sp_matrix.nnz))
    
    data_to_save = {
        'train': train_sp_matrix,
        'valid_input': valid_input_sp_matrix,
        'valid_target': valid_target_sp_matrix,
        'test_input': test_input_sp_matrix,
        'test_target': test_target_sp_matrix,
        'num_users': num_users,
        'num_items': num_items
    }

    info_to_save = {
        'user_id_dict': user_id_dict,
        'user_to_num_items': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_to_num_users': item_to_num_users
    }

    with open(file_prefix + '.stat', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    with open(file_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess hold_user_out finished.')


















def preprocess_others(data, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, test_ratio=0.2, valid_ratio=0.1, leave_k=5, split_type='holdout', split_random=True, file_prefix='data'):
    num_users = len(user_id_dict)
    num_items = len(item_id_dict)
    num_ratings = len(data)

    if split_type == 'holdout':
        pass
    elif split_type == 'loo':
        valid_ratio = 1
        test_ratio = 1
    elif split_type == 'lko':
        valid_ratio = leave_k
        test_ratio = leave_k

    train_data, test_data = split_train_test(data, test_ratio=test_ratio, split_random=split_random)
    train_data, valid_data = split_train_test(train_data, test_ratio=valid_ratio, split_random=split_random)

    # train_dict = df_to_dict(train_data)
    # valid_dict = df_to_dict(valid_data)
    # test_dict = df_to_dict(test_data)

    train_sp_matrix = df_to_sparse(train_data, shape=(num_users, num_items))
    valid_sp_matrix = df_to_sparse(valid_data, shape=(num_users, num_items))
    test_sp_matrix = df_to_sparse(test_data, shape=(num_users, num_items))

    ratings_per_user = list(user_to_num_items.values())

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f\n" % (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))

    info_lines.append('# train users: %d, # train ratings: %d' % (train_sp_matrix.shape[0], train_sp_matrix.nnz))
    info_lines.append('# valid users: %d, # valid ratings: %d' % (valid_sp_matrix.shape[0], valid_sp_matrix.nnz))
    info_lines.append('# test users: %d, # test ratings: %d' % (test_sp_matrix.shape[0], test_sp_matrix.nnz))
    
    data_to_save = {
        'train': train_sp_matrix,
        'valid': valid_sp_matrix,
        'test': test_sp_matrix,
        'num_users': num_users,
        'num_items': num_items
    }

    info_to_save = {
        'user_id_dict': user_id_dict,
        'user_to_num_items': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_to_num_users': item_to_num_users
    }

    with open(file_prefix + '.stat', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    with open(file_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess %s finished.' % split_type)

def df_to_sparse(df, shape):
    rows, cols = df.user, df.item
    values = df.ratings

    sp_data = sp.csr_matrix((values, (rows, cols)), dtype='float64', shape=shape)

    num_nonzeros = np.diff(sp_data.indptr)
    rows_to_drop = num_nonzeros == 0
    if sum(rows_to_drop) > 0:
        print('%d empty users are dropped from matrix.' % sum(rows_to_drop))
        sp_data = sp_data[num_nonzeros != 0]

    return sp_data

def df_to_dict(df):
    df_dict = {}
    for row in df.itertuples():
        if row.user not in df_dict:
            df_dict[row.user] = []
        df_dict[row.user].append((row.item, row.ratings, row.timestamps))
    return df_dict

def split_train_test(df, test_ratio=0.2, split_random=True):
    df_group = df.groupby('user')
    train_list, test_list = [], []

    for _, group in df_group:
        user = pd.unique(group.user)[0]
        num_items_user = len(group)
        # TODO
        num_test_items = int(math.ceil(test_ratio * num_items_user)) if isinstance(test_ratio, float) else test_ratio
        group = group.sort_values(by='timestamps')
        
        idx = np.ones(num_items_user, dtype='bool')
        if split_random:
            test_idx = np.random.choice(num_items_user, num_test_items, replace=False)
            idx[test_idx] = False
        else:    
            idx[-num_test_items:] = False

        train_list.append(group[idx])
        test_list.append(group[np.logical_not(idx)])
        if len(group[idx]) == 0:
            print('zero train')

        if len(group[np.logical_not(idx)]) == 0:
            print('zero test')
    
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df


def split_train_test_5_fold(df, split_random=True): # jhkim, jwlee
    df_group = df.groupby('user')

    fold_train_list = [[] for i in range(5)]
    fold_test_list = [[] for i in range(5)]

    for _, group in df_group:
        user = pd.unique(group.user)[0]
        num_items_user = len(group)

        # TODO
        num_test_items_per_fold = int(num_items_user / 5)
        group = group.sort_values(by='timestamps')

        if split_random:
            fold_idx = np.random.permutation(num_items_user)
        else:
            fold_idx = np.arange(num_items_user)

        for i in range(5):
            idx = np.ones(num_items_user, dtype='bool')
            test_index = fold_idx[i*num_test_items_per_fold : (i+1)*num_test_items_per_fold]
            idx[test_index] = False

            fold_train_list[i].append(group[idx])
            fold_test_list[i].append(group[np.logical_not(idx)])

            if len(group[idx]) == 0:
                print('zero train %d fold'%(i))

            if len(group[np.logical_not(idx)]) == 0:
                print('zero test %d fold'%(i))
    
    train_df = []
    test_df = []

    for i in range(5):
        train_df.append(pd.concat(fold_train_list[i]))
        test_df.append(pd.concat(fold_test_list[i]))

    return train_df, test_df


def preprocess_side_information(raw_file, file_prefix, data_name, separator, info_file):

    if data_name == 'ml-1m' or data_name == 'ml-10m' or data_name == 'ml-20m':
        data = pd.read_csv(raw_file, sep=separator, names=['movieId', 'genre'],
                                                dtype={'movieId': int,  'genre': str},
                                                usecols=[0, 2],
                                                engine='python')

        with open(info_file, 'rb') as f:
            info_dict = pickle.load(f)

        item_id_dict = info_dict['item_id_dict']


        num_movies = len(item_id_dict)
        
        all_genre = []
        genre_dict = {}

        movie_id_list = []
        movie_genre_list = []

        for idx, genre in data.values:
            genre_list = genre.split('|')

            if idx not in list( item_id_dict.keys() ):
                continue

            for g in genre_list:
                if g not in all_genre:
                    genre_dict[g] = len(all_genre)
                    all_genre.append(g)
                
                movie_id_list.append(item_id_dict[idx])
                movie_genre_list.append(genre_dict[g])
            
        values = np.ones_like(movie_id_list)
        num_genres = len(all_genre)
        
        shape = (num_movies, num_genres) 

        item_side_information = sp.csr_matrix((values, (movie_id_list, movie_genre_list)), dtype='float64', shape=shape)

        save_path = os.path.join(file_prefix, 'item.side')
        
        with open(save_path, 'wb') as f:
            pickle.dump({'item_side_mat': item_side_information}, f)