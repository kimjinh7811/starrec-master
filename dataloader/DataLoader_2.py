import math
import pickle

import numpy as np
import scipy.sparse as sp

def load_data_and_info(data_file, info_file, split_type, implicit=True):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    with open(info_file, 'rb') as f:
        info_dict = pickle.load(f)

    if split_type == 'user':
        train, valid_input, valid_target, test_input, test_target = \
            data_dict['train'], data_dict['valid_input'], data_dict['valid_target'], data_dict['test_input'], data_dict['test_target']
        user_id_dict, user_to_num_items, item_id_dict, item_to_num_users = \
            info_dict['user_id_dict'], info_dict['user_to_num_items'], info_dict['item_id_dict'], info_dict['item_to_num_users']

        for train_u in train:
            IRTs_user = train[train_u]
            irts = []
            for irt in IRTs_user:
                if implicit:
                    irts.append((irt[0], 1))
                else:
                    irts.append((irt[0], irt[1]))
            train[train_u] = irts

        for valid_u in valid_input:
            IRTs_user = valid_input[valid_u]
            irts = []
            for irt in IRTs_user:
                if implicit:
                    irts.append((irt[0], 1))
                else:
                    irts.append((irt[0], irt[1]))
            valid_input[valid_u] = irts

        for valid_u in valid_target:
            IRT_user = valid_target[valid_u]

            if implicit:
                irt = [(IRT_user[0], 1)]
            else:
                irt = [(IRT_user[0], IRT_user[1])]

            valid_target[valid_u] = irt

        for test_u in test_input:
            IRTs_user = test_input[test_u]
            irts = []
            for irt in IRTs_user:
                if implicit:
                    irts.append((irt[0], 1))
                else:
                    irts.append((irt[0], irt[1]))
            test_input[test_u] = irts

        for test_u in test_target:
            IRT_user = test_target[test_u]

            if implicit:
                irt = [(IRT_user[0], 1)]
            else:
                irt = [(IRT_user[0], IRT_user[1])]

            test_target[test_u] = irt

        return train, valid_input, valid_target, test_input, test_target, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users

    else:
        train, valid, test = data_dict['train'], data_dict['valid'], data_dict['test']
        user_id_dict, user_to_num_items, item_id_dict, item_to_num_users = info_dict['user_id_dict'], info_dict['user_to_num_items'], info_dict['item_id_dict'], info_dict['item_to_num_users']

        for train_u in train:
            IRTs_user = train[train_u]
            irts = []
            for irt in IRTs_user:
                if implicit:
                    irts.append((irt[0], 1))
                else:
                    irts.append((irt[0], irt[1]))
            train[train_u] = irts

        for valid_u in valid:
            IRTs_user = valid[valid_u]
            irts = []
            for irt in IRTs_user:
                if implicit:
                    irts.append((irt[0], 1))
                else:
                    irts.append((irt[0], irt[1]))
            valid[valid_u] = irts

        for test_u in test:
            IRTs_user = test[test_u]
            irts = []
            for irt in IRTs_user:
                if implicit:
                    irts.append((irt[0], 1))
                else:
                    irts.append((irt[0], irt[1]))
            test[test_u] = irts

        return train, valid, test, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users

# def save_leave_one_out(data_file, train_file, valid_file, test_file, info_file, separator, popularity_order=True):
#     """
#     Read data and split it into train, valid and test in leave-one-out manner.
#     Save preprocessed data into files.


#     :param str data_file: File path of data to read
#     :param str train_file: File path of train data to save
#     :param str valid_file: File path of valid data to save
#     :param str test_file: File path of test data to save
#     :param str info_file: File path of data information to save
#     :param str separator: String by which UIRT line is seperated
#     :param bool popularity_order: Boolean indicating if users and items should be sorted by their frequencies.

#     :return: None
#     """
#     # Read the data and reorder it by popularity.
#     num_users, num_items, num_ratings, user_id_to_cnt, item_id_to_cnt, UIRTs_per_user = order_by_popularity(data_file, separator, popularity_order)

#     num_ratings_per_user, num_ratings_per_item = {}, {}
#     user_ids, item_ids = {}, {}

#     # Assign new user_id for each user.
#     for _id, u in enumerate(user_id_to_cnt):
#         user_ids[u[0]] = _id
#         num_ratings_per_user[_id] = u[1]

#     # Assign new item_id for each item.
#     for _id, i in enumerate(item_id_to_cnt):
#         item_ids[i[0]] = _id
#         num_ratings_per_item[_id] = i[1]

#     # Convert UIRTs with new user_id and item_id.
#     for u in UIRTs_per_user.keys():
#         for UIRT in UIRTs_per_user[u]:
#             i = int(UIRT[1])
#             UIRT[0] = str(user_ids[u])
#             UIRT[1] = str(item_ids[i])

#     # Build train and test lines.
#     train_lines, valid_lines, test_lines = [], [], []
#     num_valid_users = 0
#     num_test_users = 0
#     for u in UIRTs_per_user.keys():
#         # Sort the UIRTs by the ascending order of the timestamp.
#         UIRTs_per_user[u] = sorted(UIRTs_per_user[u], key=lambda x: int(x[-1]))
#         # For valid, test dataset
#         if len(UIRTs_per_user[u]) > 5:
#             num_test_users += 1
#             test_UIRT = UIRTs_per_user[u][-1]
#             UIRTs_per_user[u].pop()
#             test_lines.append('\t'.join(test_UIRT))

#             num_valid_users += 1
#             valid_UIRT = UIRTs_per_user[u][-1]
#             UIRTs_per_user[u].pop()
#             valid_lines.append('\t'.join(valid_UIRT))
#         # For train dataset
#         for UIRT in UIRTs_per_user[u]:
#             train_lines.append('\t'.join(UIRT))

#     # Build info lines, user_idx_lines, and item_idx_lines.
#     info_lines, user_idx_lines, item_idx_lines = [], [], []
#     info_lines.append('\t'.join([str(num_users), str(num_items), str(num_ratings)]))
#     info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings/(num_users * num_items)))*100))
#     ratings_per_user = list(num_ratings_per_user.values())
#     info_lines.append("Min/Max/Avg. ratings per users : %d %d %.2f" %
#                       (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))
#     ratings_per_item = list(num_ratings_per_item.values())
#     info_lines.append("Min/Max/Avg. ratings per items : %d %d %.2f" %
#                       (min(ratings_per_item), max(ratings_per_item), np.mean(ratings_per_item)))
#     info_lines.append("Number of users in valid set : %d" % (num_valid_users))
#     info_lines.append("Number of users in test set : %d" % (num_test_users))
#     info_lines.append('User_id\tNumber of ratings')
#     for u in range(num_users):
#         info_lines.append("\t".join([str(u), str(num_ratings_per_user[u])]))
#     info_lines.append('\nItem_id\tNumber of ratings')
#     for i in range(num_items):
#         info_lines.append("\t".join([str(i), str(num_ratings_per_item[i])]))

#     user_idx_lines.append('Original_user_id\tCurrent_user_id')
#     for u, v in user_ids.items():
#         user_idx_lines.append("\t".join([str(u), str(user_ids[u])]))
#     item_idx_lines.append('Original_item_id\tCurrent_item_id')
#     for i, v in item_ids.items():
#         item_idx_lines.append("\t".join([str(i), str(item_ids[i])]))

#     # Save train, test, info, user_idx, item_idx files.
#     with open(train_file, 'w') as f:
#         f.write('\n'.join(train_lines))
#     with open(valid_file, 'w') as f:
#         f.write('\n'.join(valid_lines))
#     with open(test_file, 'w') as f:
#         f.write('\n'.join(test_lines))
#     with open(info_file, 'w') as f:
#         f.write('\n'.join(info_lines))
#     with open(info_file + '_user_id', 'w') as f:
#         f.write('\n'.join(user_idx_lines))
#     with open(info_file + '_item_id', 'w') as f:
#         f.write('\n'.join(item_idx_lines))
#     print("Save leave-one-out files.")


# def save_hold_out(data_file, train_file, valid_file, test_file, info_file, separator, split_ratio=[0.8, 0.2], valid_ratio=0.1, popularity_order=True):
#     """
#     Read data and split it into train, valid and test in holdout manner.
#     Save preprocessed data into files.

#     :param str data_file: File path of data to read
#     :param str train_file: File path of train data to save
#     :param str valid_file: File path of valid data to save
#     :param str test_file: File path of test data to save
#     :param str info_file: File path of data information to save
#     :param str separator: String by which UIRT line is seperated
#     :param list split_ratio: list of float indicating [(train + valid) ratio, test ratio]
#     :param float valid_ratio: float indicating the validation ratio from train data
#     :param bool popularity_order: Boolean indicating if users and items should be sorted by their frequencies.

#     :return: None
#     """
#     # Read the data and reorder it by popularity.
#     num_users, num_items, num_ratings, user_id_to_cnt, item_id_to_cnt, UIRTs_per_user \
#         = order_by_popularity(data_file, separator, popularity_order)

#     num_ratings_per_user, num_ratings_per_item = {}, {}
#     user_ids, item_ids = {}, {}

#     # Assign new user_id for each user.
#     for _id, u in enumerate(user_id_to_cnt):
#         user_ids[u[0]] = _id
#         num_ratings_per_user[_id] = u[1]

#     # Assign new item_id for each item.
#     for _id, i in enumerate(item_id_to_cnt):
#         item_ids[i[0]] = _id
#         num_ratings_per_item[_id] = i[1]

#     # Convert UIRTs with new user_id and item_id.
#     for u in UIRTs_per_user.keys():
#         for UIRT in UIRTs_per_user[u]:
#             i = int(UIRT[1])
#             UIRT[0] = str(user_ids[u])
#             UIRT[1] = str(item_ids[i])

#     train_lines, valid_lines, test_lines = [], [], []
#     num_valid_users = 0
#     num_test_users = 0
#     for u in UIRTs_per_user.keys():
#         # Sort the UIRTs by the descending order of the timestamp.
#         UIRTs_per_user[u] = sorted(UIRTs_per_user[u], key=lambda x: int(x[-1]))
#         # For valid, test dataset
#         num_ratings_by_user = len(UIRTs_per_user[u])
#         num_test_ratings = math.floor(float(split_ratio[1]) * num_ratings_by_user)
#         num_train_ratings = num_ratings_by_user - num_test_ratings
#         num_valid_ratings = math.floor(valid_ratio * num_train_ratings)
#         num_train_ratings -= num_valid_ratings
#         assert num_train_ratings + num_valid_ratings + num_test_ratings == num_ratings_by_user

#         if num_ratings_by_user > 5:
#             if num_test_ratings > 0:
#                 num_test_users += 1
#                 for _ in range(num_test_ratings):
#                     last_UIRT = UIRTs_per_user[u][-1]
#                     UIRTs_per_user[u].pop()
#                     test_lines.append('\t'.join(last_UIRT))
#             if num_valid_ratings > 0:
#                 num_valid_users += 1
#                 for _ in range(num_valid_ratings):
#                     last_UIRT = UIRTs_per_user[u][-1]
#                     UIRTs_per_user[u].pop()
#                     valid_lines.append('\t'.join(last_UIRT))
#         # For train dataset
#         for UIRT in UIRTs_per_user[u]:
#             train_lines.append('\t'.join(UIRT))

#     # Build info lines, user_idx_lines, and item_idx_lines.
#     info_lines, user_idx_lines, item_idx_lines = [], [], []
#     info_lines.append('\t'.join([str(num_users), str(num_items), str(num_ratings)]))
#     info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
#     ratings_per_user = list(num_ratings_per_user.values())
#     info_lines.append("Min/Max/Avg. ratings per users : %d %d %.2f" %
#                       (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))
#     ratings_per_item = list(num_ratings_per_item.values())
#     info_lines.append("Min/Max/Avg. ratings per items : %d %d %.2f" %
#                       (min(ratings_per_item), max(ratings_per_item), np.mean(ratings_per_item)))
#     info_lines.append("Number of users in valid set : %d" % (num_valid_users))
#     info_lines.append("Number of users in test set : %d" % (num_test_users))
#     info_lines.append('User_id\tNumber of ratings')
#     for u in range(num_users):
#         info_lines.append("\t".join([str(u), str(num_ratings_per_user[u])]))
#     info_lines.append('\nItem_id\tNumber of ratings')
#     for i in range(num_items):
#         info_lines.append("\t".join([str(i), str(num_ratings_per_item[i])]))

#     user_idx_lines.append('Original_user_id\tCurrent_user_id')
#     for u, v in user_ids:
#         user_idx_lines.append("\t".join([str(u), str(user_ids[u])]))
#     item_idx_lines.append('Original_item_id\tCurrent_item_id')
#     for i, v in item_ids:
#         item_idx_lines.append("\t".join([str(i), str(item_ids[i])]))

#     # Save train, test, info, user_idx, item_idx files.
#     with open(train_file, 'w') as f:
#         f.write('\n'.join(train_lines))
#     with open(valid_file, 'w') as f:
#         f.write('\n'.join(valid_lines))
#     with open(test_file, 'w') as f:
#         f.write('\n'.join(test_lines))
#     with open(info_file, 'w') as f:
#         f.write('\n'.join(info_lines))
#     with open(info_file + '_user_id', 'w') as f:
#         f.write('\n'.join(user_idx_lines))
#     with open(info_file + '_item_id', 'w') as f:
#         f.write('\n'.join(item_idx_lines))
#     print("Save hold-one-out files.")


# def order_by_popularity(data_file, separator, popularity_order=True):
#     """
#     Reads data file.
#     Returns list of item-rating-time per users, with statistics such as number of users/items etc.

#     :param str data_file: Filepath to read
#     :param str separator: String by which UIRT line is seperated
#     :param bool popularity_order: Boolean indicating if users and items should be sorted by their frequencies.

#     :return int num_users: Number of users
#     :return int num_items: Number of items
#     :return int num_ratings: Number of ratings
#     :return list user_id_to_cnt: list of (user id, frequency) tuples
#     :return list item_id_to_cnt: list of (item id, frequency) tuples
#     :return dictionary UIRTs_per_user: Dictionary which maps user id to its list of IRT
#     """
#     num_users, num_items, num_ratings = 0, 0, 0
#     user_id_to_cnt, item_id_to_cnt, UIRTs_per_user = {}, {}, {}

#     # Read the data file.
#     print("Loading the dataset from \"%s\"" % data_file)
#     with open(data_file, "r") as f:
#         # Format (user_id, item_id, rating, timestamp)
#         for line in f.readlines():
#             user_id, item_id, rating, time = line.strip().split(separator)
#             user_id, item_id = int(user_id), int(item_id)

#             # Update the number of ratings per user
#             if user_id not in user_id_to_cnt:
#                 user_id_to_cnt[user_id] = 1
#                 UIRTs_per_user[user_id] = []
#                 num_users += 1
#             else:
#                 user_id_to_cnt[user_id] += 1

#             # Update the number of ratings per item
#             if item_id not in item_id_to_cnt:
#                 item_id_to_cnt[item_id] = 1
#                 num_items += 1
#             else:
#                 item_id_to_cnt[item_id] += 1

#             num_ratings += 1
#             line = [str(user_id), str(item_id), str(rating), str(time)]
#             UIRTs_per_user[user_id].append(line)
#     print("\"num_users\": %d, \"num_items\": %d, \"num_ratings\": %d" % (num_users, num_items, num_ratings))

#     if popularity_order:
#         # Sort the user_ids and item_ids by the popularity
#         user_id_to_cnt = sorted(user_id_to_cnt.items(), key=lambda x: x[-1], reverse=True)
#         item_id_to_cnt = sorted(item_id_to_cnt.items(), key=lambda x: x[-1], reverse=True)
#     else:
#         user_id_to_cnt = user_id_to_cnt.items()
#         item_id_to_cnt = item_id_to_cnt.items()

#     return num_users, num_items, num_ratings, user_id_to_cnt, item_id_to_cnt, UIRTs_per_user

# def read_data_file(train_file, valid_file, test_file, info_file, implicit):
#     """
#     Reads data files which are already splitted and saved.
#     Returns {train, valid, test} matrices, dictionary of train data with number of users and items.

#     :param str train_file: Filepath of train data
#     :param str valid_file: Filepath of valid data
#     :param str test_file: Filepath of test data
#     :param str info_file: Filepath of data info
#     :param bool implicit: Boolean indicating if rating should be converted to 1

#     :return int num_users: Number of users
#     :return int num_items: Number of items
#     :return dict train_dict: Dictionary of training data. Key: Value = User id: List of related items.
#     :return dict valid_dict: Dictionary of valid data. Key: Value = User id: List of related items.
#     :return dict test_dict: Dictionary of test data. Key: Value = User id: List of related items.
#     """

#     # Read the meta file.
#     separator = '\t'
#     with open(info_file, "r") as f:
#         # The first line is the basic information for the dataset.
#         num_users, num_items, num_ratings = list(map(int, f.readline().split(separator)))

#     # Build training and test matrices.
#     train_dict = {u: [] for u in range(num_users)}
#     valid_dict = {u: [] for u in range(num_users)}
#     test_dict = {u: [] for u in range(num_users)}

#     # Read the training file.
#     print("Loading the train data from \"%s\"" % train_file)
#     with open(train_file, "r") as f:
#         for line in f.readlines():
#             u, i, r, t = line.strip().split(separator)
#             user_id, item_id, rating, time = int(u), int(i), float(r), int(t)
#             if implicit:
#                 rating = 1
#             train_dict[user_id].append([item_id, rating])

#     # Read the valid file.
#     print("Loading the valid data from \"%s\"" % valid_file)
#     with open(valid_file, "r") as f:
#         for line in f.readlines():
#             u, i, r, t = line.strip().split(separator)
#             user_id, item_id, rating, time = int(u), int(i), float(r), int(t)
#             if implicit:
#                 rating = 1
#             valid_dict[user_id].append([item_id, rating])

#     # Read the test file.
#     print("Loading the test data from \"%s\"" % test_file)
#     with open(test_file, "r") as f:
#         for line in f.readlines():
#             u, i, r, t = line.strip().split(separator)
#             user_id, item_id, rating, time = int(u), int(i), float(r), int(t)
#             if implicit:
#                 rating = 1
#             test_dict[user_id].append([item_id, rating])

#     print("\"num_users\": %d, \"num_items\": %d, \"num_ratings\": %d" % (num_users, num_items, num_ratings))
#     return num_users, num_items, train_dict, valid_dict, test_dict