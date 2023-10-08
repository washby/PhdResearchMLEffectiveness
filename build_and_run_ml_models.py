import json
import logging

import pandas as pd
import database
import pickle
from copy import deepcopy
import utils
import datetime

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=fr'logs\ml_{datetime.datetime.now()}.log', filemode='w')

    header_list = ["user_id", "course_id", "total_activity_time", "user_state", "term_id", "crs_count", "avg_crs_level",
                   "avg_crs_credits", "grade", "adjusted_grade", "submission_day_diff", "pg_level", "pg_normed",
                   "pt_level", "pt_normed", "late_percent", "on_time_percent", "missing_percent", "FAIL"]


    with open('data_config.json') as f:
        data_config = json.load(f)

    all_train_data = pd.DataFrame()
    all_test_data = pd.DataFrame()

    for campus in data_config:
        with(open(f'{campus["name"]}.pkl', 'rb')) as f:
            df = pickle.load(f)

        df = [list(item) for item in df]
        df = pd.DataFrame(df, columns=header_list)

        train_data, test_data = utils.build_test_and_train_data(df, campus['term_ids'])

        logging.info(f'{campus["name"]} has {len(train_data)} train records and {len(test_data)} test records')
        all_train_data = pd.concat([all_train_data, train_data], ignore_index=True)
        all_test_data = pd.concat([all_test_data, test_data], ignore_index=True)

    fails = all_train_data['FAIL'].tolist() + all_test_data['FAIL'].tolist()
    fails_percentage = sum(fails) / len(fails)
    print(f'fails % = {fails_percentage}')

    logging.info(f'All campuses have {all_train_data.shape} train records and {all_test_data.shape} test records')

    logging.info("Beginning Naive Bayes")
    nb_results = utils.run_naive_bayes(deepcopy(all_train_data), deepcopy(all_test_data))
    logging.info("Writing Naive Bayes results to file")
    pd.DataFrame(nb_results).to_csv('nb_results.csv', index=False)

    logging.info("Beginning Decision Tree")
    tree_results = utils.run_decision_tree(deepcopy(all_train_data), deepcopy(all_test_data))
    logging.info("Writing Decision Trees results to file")
    pd.DataFrame(tree_results).to_csv('tree_results.csv', index=False)

    logging.info("Beginning Neural Network")
    nn_results = utils.run_neural_network(deepcopy(all_train_data), deepcopy(all_test_data))
    logging.info("Writing Neural Network results to file")
    pd.DataFrame(nn_results).to_csv('nn_results.csv', index=False)

    logging.info("Beginning SVM")
    svm_results = utils.run_SVM(deepcopy(all_train_data), deepcopy(all_test_data))
    logging.info("Writing SVM results to file")
    pd.DataFrame(svm_results).to_csv('svm_results.csv', index=False)
