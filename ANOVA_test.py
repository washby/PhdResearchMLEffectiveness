import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


if __name__ == "__main__":
    nb_df = pd.read_csv('nb_results.csv')
    tree_df = pd.read_csv('tree_results.csv')
    nn_df = pd.read_csv('nn_results.csv')
    svm_df = pd.read_csv('svm_results.csv')

    nb_f1 = nb_df['f1_measure'].tolist()
    tree_f1 = tree_df['f1_measure'].tolist()
    nn_f1 = nn_df['f1_measure'].tolist()
    svm_f1 = svm_df['f1_measure'].tolist()

    print(f'nb_f1: {len(nb_f1)}')
    print(f'tree_f1: {len(tree_f1)}')
    print(f'nn_f1: {len(nn_f1)}')
    print(f'svm_f1: {len(svm_f1)}')
    print(f'nb_f1 avg: {np.mean(nb_f1)}')
    print(f'tree_f1 avg: {np.mean(tree_f1)}')
    print(f'nn_f1 avg: {np.mean(nn_f1)}')
    print(f'svm_f1 avg: {np.mean(svm_f1)}')

    base_line = [0.5] * len(nb_f1)

    group_names = ['base_line', 'nb_f1', 'tree_f1', 'nn_f1', 'svm_f1']
    groups = [base_line, nb_f1, tree_f1, nn_f1, svm_f1]
    groups_dict = dict(zip(group_names, groups))

    # 1. Compute the grand mean
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # 2. Compute SS_total
    SS_total = np.sum((all_data - grand_mean) ** 2)

    # 3. Compute SS_within
    SS_within = sum([(len(group) - 1) * np.var(group, ddof=1) for group in groups])

    # 4. Compute SS_between
    SS_between = SS_total - SS_within

    # 5. Calculate Cohen's f
    f_effect_size = np.sqrt(SS_between / SS_within)

    print("Cohen's f effect size:", f_effect_size)


    f_stat, p_value = f_oneway(*groups)

    print(f'f_stat: {f_stat}')
    print(f'p_value: {p_value}')

    if p_value < 0.05:
        data = np.concatenate(groups)
        labels = np.concatenate([[name] * len(group) for name, group in zip(group_names, groups)])
        results = pairwise_tukeyhsd(data, labels)
        print(results)