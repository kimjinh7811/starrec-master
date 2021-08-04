# Config
**StarRec** uses its own implementation to handle configurations of experiments (`utils/Config.py`) with `.cfg` file as extention. Config has two types: *`main_config.cfg`* in base directory and *`[model].cfg`* in `model_config`. 

The former deals with all settings related to experiment 'itself' such as the directory the dataset are stored, dataset split, early stop etc. The latter deals with hyper-parameters of models to tune such as hidden dimension, learning rate. 

Below are the descriptions and choice options of each value in *`main_config.cfg`.

| Config | Description | Option |
|-------|----------|----|
|`Dataset:data_dir`|base directory of data|-|
|`Dataset:dataset`|Name of dataset to use|ml-100k, ml-1m|
|`Dataset:min_user_per_item`|Minimum number of users for each item to have <br/> (Item below this number deleted)|-|
|`Dataset:min_item_per_user`|Minimum number of items for each user to have <br/> (User below this number deleted)|-|
|`Dataset:binarize_threshold`|Threshold that ratings equal to or greater than are binarized|-|
|`Dataset:split_type`|Type of data split|holdout, loo, lko, hold-user-out|
|`Dataset:split_random`|Split ratings randomly if true, otherwise split from recent|true, false|
|`Dataset:test_ratio`|Ratio of test data from full data|-|
|`Dataset:valid_ratio`|Ratio of validation data from training data|-|
|`Dataset:leave_k`|# of rating to leave out for evaluation (k) in leave-k-out (lko)|-|
|`Dataset:holdout_users`|Proportion of users to leave out for evaluation in hold-user-out|-|
|`Dataset:eval_neg_num`|# of negative items to consider in evaluation. <br/> Consider all negative items if 0|-|
|`Evaluator:top_k`|List of 'k's to cut off recommendation|-|
|`EarlyStop:early_stop`|# of epochs to endure with no improvement of measure|-|
|`EarlyStop:early_stop_measure`|Measure to optimize||
|`Experiment:num_epochs`|# of epochs|-|
|`Experiment:verbose`|If > 0, print batch loss with this value as an interval. Do not print if 0|-|
|`Experiment:print_step`|Interval of epoch to print loss and valid scores if exist|-|
|`Experiment:test_step`|Interval of epoch to evaluate on validation data|-|
|`Experiment:test_from`|Epoch from which validation begins|-|
|`Experiment:task`|Recommendation task| Ranking (currently available) |
|`Experiment:model_name`|Model to experiment| See `model` directory for more details|
|`Experiment:seed`|Random seed |-|
|`Experiment:param_search`|Search over hyper-parameter search space if true. Otherwise, run single experiment.| true, false |
|`Experiment:search_method`|Hyper-parameter search algorithm| grid, bayesian |