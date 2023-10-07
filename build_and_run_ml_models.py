import json
import logging

import pandas as pd
import database
import pickle
from copy import deepcopy
import utils

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='ml.log', filemode='w')

    header_list = ["user_id", "course_id", "total_activity_time", "user_state", "term_id", "crs_count", "avg_crs_level",
                   "avg_crs_credits", "grade", "adjusted_grade", "submission_day_diff", "pg_level", "pg_normed",
                   "pt_level", "pt_normed", "late_percent", "on_time_percent", "missing_percent", "FAIL"]

    with(open('pickled_df.pkl', 'rb')) as f:
        df = pickle.load(f)

    df = [list(item) for item in df]
    df = pd.DataFrame(df, columns=header_list)

    train_data, test_data = utils.build_test_and_train_data(df, [128])

    logging.info("train_data shape: {}".format(train_data.shape))
    logging.info("test_data shape: {}".format(test_data.shape))


    logging.info("Beginning SVM")
    svm_results = utils.run_SVM(deepcopy(train_data), deepcopy(test_data))
    logging.info("Writing SVM results to file")
    pd.DataFrame(svm_results).to_csv('nn_results.csv', index=False)
    exit(-42)

    logging.info("Beginning Naive Bayes")
    nb_results = utils.run_naive_bayes(deepcopy(train_data), deepcopy(test_data))
    logging.info("Writing Naive Bayes results to file")
    pd.DataFrame(nb_results).to_csv('nb_results.csv', index=False)

    logging.info("Beginning Decision Tree")
    tree_results = utils.run_decision_tree(deepcopy(train_data), deepcopy(test_data))
    logging.info("Writing Decision Trees results to file")
    pd.DataFrame(tree_results).to_csv('tree_results.csv', index=False)

    logging.info("Beginning Neural Network")
    nn_results = utils.run_neural_network(deepcopy(train_data), deepcopy(test_data))
    logging.info("Writing Neural Network results to file")
    pd.DataFrame(nn_results).to_csv('nn_results.csv', index=False)
