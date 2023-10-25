from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from os.path import join

import pandas as pd
from scipy.stats import levene, f_oneway, stats, f, ttest_1samp
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.multitest import multipletests

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

    def __str__(self):
        return f'DataCompiler with {len(self.campus_data)} campuses.'

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
            df['PASS'], df['FAIL'] = df['FAIL'], df['PASS']

            campus_train_data, campus_test_data = utils.build_test_and_train_data(df, campus['term_ids'])
            self.campus_data.append(Campus(campus['name'], campus_train_data, campus_test_data))

            self.all_train_data = pd.concat([self.all_train_data, campus_train_data], ignore_index=True)
            self.all_test_data = pd.concat([self.all_test_data, campus_test_data], ignore_index=True)

    def counts_table_by_campus(self):
        result = f'Campus\tTrain\tTest\tSplit\tTrain CRS\tTest CRS\tSplit\n'
        for campus in self.campus_data:
            campus_train_split = len(campus.train_data)/(len(campus.train_data) + len(campus.test_data))
            crs_train_split = len(campus.train_data_by_crs)/(len(campus.train_data_by_crs) + len(campus.test_data_by_crs))
            campus_split = f'{round(campus_train_split*100, 2)}%/{round((1-campus_train_split)*100, 2)}%'
            crs_split = f'{round(crs_train_split*100, 2)}%/{round((1-crs_train_split)*100, 2)}%'
            result += f'{campus.name}\t{len(campus.train_data)}\t{len(campus.test_data)}\t{campus_split}\t'
            result += f'{len(campus.train_data_by_crs)}\t{len(campus.test_data_by_crs)}\t{crs_split}\n'
        return result

    def fail_percents_table_by_campus(self):
        result = f'Campus\tTrain Fail Count\tTrain Total Count\tTrain Fail %\t'
        result += f'Test Fail Count\tTest Total Count\tTest Fail %\n'
        for campus in self.campus_data:
            train_fail_count = campus.train_data["FAIL"].sum()
            test_fail_count = campus.test_data["FAIL"].sum()
            train_len = len(campus.train_data['FAIL'])
            test_len = len(campus.test_data['FAIL'])
            train_fail_percent = train_fail_count / train_len
            test_fail_percent = test_fail_count / test_len
            result += f'{campus.name}\t{train_fail_count}\t{train_len}\t{train_fail_percent}\t'
            result += f'{test_fail_count}\t{test_len}\t{test_fail_percent}\n'
        return result

    def measurement_stats_table(self, measurement):
        result = f'Model\t{measurement} Mean\tMode\t{measurement} Min\t{measurement} Max\t'
        result += f'1st Quartile\tMedian\t3rd Quartile\t{measurement} Stdev\n'
        for model in self.__model_info:
            model_name, file_name = model
            file_path = join(self.__results_dir, file_name)
            df = pd.read_csv(file_path)
            result += f'{model_name}\t{df[measurement].mean()}\t{df[measurement].mode()[0]}\t'
            result += f'{df[measurement].min()}\t{df[measurement].max()}\t'
            result += f'{df[measurement].quantile(0.25)}\t{df[measurement].median()}\t'
            result += f'{df[measurement].quantile(0.75)}\t{df[measurement].std()}\n'
        return result

    def create_histogram(self, measurement, title, save=False):
        # Sample data (replace this with your data)

        for model in self.__model_info:
            model_name, file_name = model
            file_path = join(self.__results_dir, file_name)
            df = pd.read_csv(file_path)
            # Create a histogram
            n, bins, patches = plt.hist(df[measurement], bins=np.arange(0, 1.1, 0.1), edgecolor='black')
            plt.title('')
            plt.xlabel(f'{title}')
            plt.ylabel('Frequency')
            plt.grid(False)
            for bin_value, patch in zip(n, patches):
                height = patch.get_height()-20
                plt.annotate(f'{int(bin_value)}', xy=(patch.get_x() + patch.get_width() / 2, height), xytext=(0, 5),
                             textcoords='offset points', ha='center', va='bottom')

            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            if save:
                plt.savefig(join('results', f'{model_name}_{measurement}_histogram.png'))
            plt.show()

    def bonferroni_correction(self, group_list, alpha=0.05):
        n = len(group_list)
        print(f'There are {n} groups')
        comparisons = n * (n - 1) / 2
        adjusted_alpha = alpha / comparisons

        significant_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                t_stat, p_val = stats.ttest_ind(group_list[i], group_list[j])
                if p_val < adjusted_alpha:
                    significant_pairs.append((i, j, p_val))

        return significant_pairs

    def scheffes_test(self, groups):
        k = len(groups)

        # Calculate group means and overall mean
        group_means = [np.mean(group) for group in groups]
        overall_mean = np.mean([val for group in groups for val in group])

        # Calculate between-group and within-group sum of squares
        SS_between = sum([len(group) * (mean - overall_mean) ** 2 for group, mean in zip(groups, group_means)])
        SS_within = sum([sum([(val - mean) ** 2 for val in group]) for group, mean in zip(groups, group_means)])

        # Calculate degrees of freedom and mean squares
        df_between = k - 1
        df_within = len([val for group in groups for val in group]) - k
        MS_between = SS_between / df_between
        MS_within = SS_within / df_within

        # Calculate Scheffé's criterion
        n = len(groups[0])  # Assuming equal sample sizes
        S = np.sqrt(MS_within / n * df_between)

        significant_pairs = []

        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(group_means[i] - group_means[j])
                F = diff ** 2 / MS_within
                critical_value = f.ppf(0.95, df_between, df_within)

                if F > critical_value:
                    significant_pairs.append((i, j))

        return significant_pairs

    def one_way_anova(self, data):
        f_value, p_value = stats.f_oneway(*data)

        # Calculate degrees of freedom
        n_groups = len(data)
        n_total = sum(len(group) for group in data)
        df_between = n_groups - 1
        df_within = n_total - n_groups
        df_total = n_total - 1

        # Calculate sum of squares
        overall_mean = np.mean(np.concatenate(data))
        ss_total = np.sum(np.square(np.concatenate(data) - overall_mean))
        ss_between = sum(len(group) * np.square(np.mean(group) - overall_mean) for group in data)
        ss_within = ss_total - ss_between

        # Calculate mean square
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        print(f'F1-Value\tSum of Squares\tDegrees of Freedom\tMean Square\tF-Statistic\tP-Value')
        print(f'Between Groups\t{ss_between}\t{df_between}\t{ms_between}\t{f_value}\t{p_value}')
        print(f'Within Groups\t{ss_within}\t{df_within}\t{ms_within}')
        print(f'Total\t{ss_total}\t{df_total}')

        return f_value, p_value, df_total, ms_between, ms_within


    def one_sample_t_test(self, measurement):
        data_with, _, _, names = self.build_model_f1_data_sets(measurement, include_names=True)

        print("Model\tT-Statistic\tP-Value\tCohen's d")
        for i, data in enumerate(data_with):
            t_stat, p_value = ttest_1samp(data, 0.5)

            # Cohen's d calculation
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)
            cohen_d = (sample_mean - 0.5) / sample_std

            print(f'{names[i]}\t{t_stat:.3f}\t{p_value:.3f}\t{cohen_d:.3f}')

    def run_anova_test_info(self, measurement):
        data_with, data_without, labels_with = self.build_model_f1_data_sets(measurement)

        f_value, p_value, df_total, ms_between, ms_within  = self.one_way_anova(data_with)
        print(f'WITH - One-way ANOVA for {measurement}: {f_value}, {p_value}')
        f_value, p_value, df_total, ms_between, ms_within  = self.one_way_anova(data_without)
        print(f'WITHOUT - One-way ANOVA for {measurement}: {f_value}, {p_value}')

        lev_stat, lev_p = levene(*data_with, center='mean')
        print(f'Levene\'s test for {measurement}: {lev_stat}, {lev_p}')
        lev_stat, lev_p = levene(*data_without, center='mean')
        print(f'Levene\'s test for {measurement}: without Naive Bayes {lev_stat}, {lev_p}')

        mc = MultiComparison(np.concatenate(data_with), np.concatenate(labels_with))

        # Perform Tukey's HSD Test
        result_tukey = mc.tukeyhsd()

        print("\nTukey's HSD Test:")
        print(result_tukey)

    def build_model_f1_data_sets(self, measurement, include_names=False):
        name_list = []
        data_with = []
        data_without = []
        labels_with = []
        labels_without = []
        for model in self.__model_info:
            model_name, file_name = model
            name_list.append(model_name)
            file_path = join(self.__results_dir, file_name)
            df = pd.read_csv(file_path)
            labels = [model_name for _ in range(len(df[measurement]))]
            labels_with.append(labels)
            data_with.append(df[measurement])
            if model_name == 'Naive Bayes':
                continue
            labels_without.append(labels)
            data_without.append(df[measurement])
        if include_names:
            return data_with, data_without, labels_with, name_list
        return data_with, data_without, labels_with




