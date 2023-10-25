import logging
from os.path import join

import pandas as pd
from copy import deepcopy
import utils
import datetime

from utils import build_all_campus_dataframe

if __name__ == '__main__':

    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=fr'logs\ml_{date_str}.log', filemode='w')


    all_train_data, all_test_data = build_all_campus_dataframe()

    results_dir = 'results'

    fails = all_train_data['FAIL'].tolist() + all_test_data['FAIL'].tolist()
    fails_percentage = sum(fails) / len(fails)
    print(f'fails % = {fails_percentage}')

    logging.info(f'All campuses have {all_train_data.shape} train records and {all_test_data.shape} test records')

    logging.info("Beginning Naive Bayes")
    nb_results = utils.build_and_run_naive_bayes(deepcopy(all_train_data), deepcopy(all_test_data))
    logging.info("Writing Naive Bayes results to file")
    pd.DataFrame(nb_results).to_csv(join(results_dir, 'best_nb_results.csv'), index=False)

    logging.info("Beginning Decision Tree")
    tree_results = utils.build_and_run_decision_tree(deepcopy(all_train_data), deepcopy(all_test_data))
    logging.info("Writing Decision Trees results to file")
    pd.DataFrame(tree_results).to_csv(join(results_dir, 'best_tree_results.csv'), index=False)

    exit(-24)

    logging.info("Beginning Neural Network")
    nn_results = utils.build_and_run_neural_network(deepcopy(all_train_data), deepcopy(all_test_data))
    logging.info("Writing Neural Network results to file")
    pd.DataFrame(nn_results).to_csv(join(results_dir,'best_mlp_results.csv'), index=False)

    logging.info("Beginning SVM")
    svm_results = utils.build_and_run_SVM(deepcopy(all_train_data), deepcopy(all_test_data))
    logging.info("Writing SVM results to file")
    pd.DataFrame(svm_results).to_csv(join(results_dir,'best_svm_results.csv'), index=False)
