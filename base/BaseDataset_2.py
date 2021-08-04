import os
# from dataloader.DataLoader import save_hold_out, save_leave_one_out
from dataloader.Preprocess import preprocess
import utils.Constant as CONSTANT

class BaseDataset:
    def __init__(self, data_dir, dataset, implicit, split_type="loo", train_ratio=0.8, valid_ratio=0.1, split_random=True, popularity_order=True):
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
        self.split_type = split_type
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.split_random = split_random
        self.popularity_order = popularity_order
        self.min_item_cnt = 10

        file_prefix = os.path.join(self.data_dir, self.data_name, self.data_name)
        self.raw_file = file_prefix + '.rating'
        self.file_prefix = os.path.join(self.data_dir, self.data_name, self.data_name) + '_' + self.split_type
        if self.split_type == 'holdout':
            self.file_prefix += '_%.1f_%.1f' % (self.train_ratio, self.valid_ratio)
        self.data_file = self.file_prefix + '.data'
        self.info_file = self.file_prefix + '.info'
        self.separator = CONSTANT.DATASET_TO_SEPRATOR[dataset]

        if not self.check_dataset_exists():
            preprocess(self.raw_file, self.file_prefix, self.split_type, self.train_ratio, self.valid_ratio, self.split_random, self.min_item_cnt, self.separator, self.popularity_order)

    def check_dataset_exists(self):
        # info_file = os.path.join(self.data_dir, self.data_name, self.data_name) + '.info'
        # data_file = os.path.join(self.data_dir, self.data_name, self.data_name) + '.data'
        return os.path.exists(self.data_file) and os.path.exists(self.info_file)

    def __str__(self):
        return 'BaseDataset'