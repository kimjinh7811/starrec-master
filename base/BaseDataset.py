import os
from dataloader.Preprocess import preprocess
from dataloader.Preprocess import preprocess_side_information
import utils.Constant as CONSTANT

class BaseDataset:
    def __init__(self, data_dir, dataset, min_user_per_item=1, min_item_per_user=1, implicit=True, binarize_threshold=1.0, split_type="loo", split_random=True,
                                test_ratio=0.2, valid_ratio=0.1, leave_k=5, holdout_users=100, popularity_order=True):
        """
        Dataset class

        :param str data_dir: base directory of data
        :param str dataset: Name of dataset e.g. ml-100k
        :param str separator: String by which UIRT line is seperated
        :param bool implicit: Boolean indicating if rating should be converted to 1
        :param str split_type: Type of data split. leave-one-out (loo) or holdout
        :param list split_ratio: list of float indicating [(train + valid) ratio, test ratio]
        :param float valid_ratio: float indicating the validation ratio from train data
        :param bool popularity_order: Boolean indicating if users and items should be sorted by their frequencies.
        """

        self.data_dir = data_dir
        self.data_name = dataset

        self.min_user_per_item = min_user_per_item
        self.min_item_per_user = min_item_per_user

        self.implicit = implicit
        self.binarize_threshold = binarize_threshold

        self.split_type = split_type
        self.split_random = split_random

        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio        
        self.leave_k = leave_k
        self.holdout_users = holdout_users
        self.popularity_order = popularity_order

        prefix = os.path.join(self.data_dir, self.data_name, self.data_name)

        if self.data_name == 'ml-1m' or self.data_name == 'ml-10m':
            self.raw_file = os.path.join(prefix, 'ratings.dat')
            self.raw_side_inform_file = os.path.join('./data',self.data_name, self.data_name, 'movies.dat')
        elif self.data_name == 'ml-20m':
            self.raw_file = os.path.join(prefix, 'ratings.csv')
            self.raw_side_inform_file = os.path.join('./data',self.data_name, self.data_name, 'movies.csv')

        self.raw_file = prefix + '.rating'
        


        self.file_prefix = self.generate_file_prefix()
        self.data_file = self.file_prefix + '.data'
        self.info_file = self.file_prefix + '.info'

        self.side_inform_file_prefix = os.path.join('./data', self.data_name)
        self.side_information = os.path.join(self.side_inform_file_prefix, 'item.side')

        self.separator = CONSTANT.DATASET_TO_SEPRATOR[dataset]

        if not self.check_dataset_exists():
            print('preprocess raw data...')
            preprocess(self.raw_file, self.file_prefix, self.split_type, self.split_random, self.test_ratio, self.valid_ratio, 
                        self.holdout_users, self.leave_k, self.implicit, self.binarize_threshold, self.min_item_per_user, self.min_user_per_item, self.separator, self.popularity_order)
        
        # if not self.check_side_information_exists():
        #     print('preprocss side_information...')
        #     #preprocess_side_information(self.raw_side_inform_file, self.side_inform_file_prefix, self.data_name, self.separator, self.info_file)

            
        #     preprocess(self.raw_file, self.file_prefix, self.split_type, self.test_ratio, self.valid_ratio, self.split_random, self.min_item_cnt, self.separator, self.popularity_order)

    def check_dataset_exists(self):
        # info_file = os.path.join(self.data_dir, self.data_name, self.data_name) + '.info'
        # data_file = os.path.join(self.data_dir, self.data_name, self.data_name) + '.data'
        return os.path.exists(self.data_file) and os.path.exists(self.info_file)

    def check_side_information_exists(self):
        # info_file = os.path.join(self.data_dir, self.data_name, self.data_name) + '.info'
        # data_file = os.path.join(self.data_dir, self.data_name, self.data_name) + '.data'
        return os.path.exists(self.side_information)

    def generate_file_prefix(self):
        sub_directory = [self.split_type]
        sub_directory.append('_mincnt_%d_%d' % (self.min_user_per_item, self.min_item_per_user))
        
        if self.implicit:
                sub_directory.append('_implicit_%.1f' % self.binarize_threshold)
        sub_directory.append('_random' if self.split_random else '_time')
        
        if self.split_type == 'holdout':
            sub_directory.append('_%.1f_%.1f' % (self.test_ratio, self.valid_ratio))
        elif self.split_type == 'hold-user-out':
            sub_directory.append('_ho_user_%d' % self.holdout_users)
        elif self.split_type == 'hold-out-user-5-fold-test': # jhkim, jwlee
            sub_directory.append('_ho_user_5f_te_%d' % self.holdout_users)
        elif self.split_type == 'lko':
            sub_directory.append('_k_%d' % self.leave_k)
        else:
            pass
        
        sub_directory = os.path.join(self.data_dir, self.data_name, '_'.join(sub_directory))
        if not os.path.exists(sub_directory):
            os.mkdir(sub_directory)
            
        return os.path.join(sub_directory, 'data')

    def __str__(self):
        return 'BaseDataset'