import json
import logging
import pickle
from os.path import join

import pandas as pd

import utils

class Campus:
    def __init__(self, name, train_data, test_data):
        self.name = name
        self.train_data = train_data
        self.test_data = test_data
        self.train_data_by_crs = train_data.groupby('course_id')
        self.test_data_by_crs = test_data.groupby('course_id')


class DataCompiler:
    def __init__(self, results_dir='results'):
        self.all_train_data = None
        self.all_test_data = None
        self.campus_data = []
        self.__build_all_data()
        self.__results_dir = results_dir
        self.__model_info = [('Naive Bayes', 'best_nb_results.csv'), ('Decision Tree', 'best_tree_results.csv'),
                             ('Neural Network', 'best_mlp_results.csv'), ('SVM', 'best_svm_results.csv')]

    def model_stats_table(self):
        result = f'Model\tF1 Measure\tMy F1 Measure\tAccuracy\n'
        for model in self.__model_info:
            model_name, file_name = model
            file_path = join(self.__results_dir, file_name)
            df = pd.read_csv(file_path)
            result += f'{model_name}\t{df["f1_measure"].mean()}\t{df["accuracy"].mean()}\n'
        return result

    def __build_all_data(self):
        header_list = ["user_id", "course_id", "total_activity_time", "user_state", "term_id", "crs_count", "avg_crs_level",
                       "avg_crs_credits", "grade", "adjusted_grade", "submission_day_diff", "pg_level", "pg_normed",
                       "pt_level", "pt_normed", "late_percent", "on_time_percent", "missing_percent", "FAIL", "PASS"]

        with open('data_config.json') as f:
            data_config = json.load(f)
        self.all_train_data = pd.DataFrame()
        self.all_test_data = pd.DataFrame()

        for campus in data_config:
            with(open(f'{campus["name"]}.pkl', 'rb')) as f:
                df = pickle.load(f)

            df = [list(item) for item in df]
            df = pd.DataFrame(df, columns=header_list)

            campus_train_data, campus_test_data = utils.build_test_and_train_data(df, campus['term_ids'])
            self.campus_data.append(Campus(campus['name'], campus_train_data, campus_test_data))

            self.all_train_data = pd.concat([self.all_train_data, campus_train_data], ignore_index=True)
            self.all_test_data = pd.concat([self.all_test_data, campus_test_data], ignore_index=True)

    def stats_table_by_campus(self):
        result = f'Campus\tTrain\tTest\tTrain CRS\tTest CRS\n'
        for campus in self.campus_data:
            result += f'{campus.name}\t{len(campus.train_data)}\t{len(campus.test_data)}\t'
            result += f'{len(campus.train_data_by_crs)}\t{len(campus.test_data_by_crs)}\n'
        return result
