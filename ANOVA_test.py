import numpy as np
import pandas as pd
from scipy.stats import f_oneway


def test_mean(values):
    """
    Run a one-way ANOVA to test if the mean of values is statistically
    different from 0.5.

    Parameters:
    - values: list of float values between 0 and 1

    Returns:
    - tuple: F-statistic, p-value
    """
    # Create the comparison group of 0.5s
    comparison_group = [0.5] * len(values)

    # Run one-way ANOVA
    f_stat, p_value = f_oneway(values, comparison_group)

    return f_stat, p_value


if __name__ == "__main__":
    nb_df = pd.read_csv('nb_results.csv')
    tree_df = pd.read_csv('tree_results.csv')
    nn_df = pd.read_csv('nn_results.csv')

    nb_f1 = nb_df['f1_measure'].tolist()
    tree_f1 = tree_df['f1_measure'].tolist()
    # nn_f1 = nn_df['f1_measure'].tolist()
    print(f'nb_f1: {len(nb_f1)}')
    print(f'tree_f1: {len(tree_f1)}')
    # print(f'nn_f1: {len(nn_f1)}')

    # values = [0.6, 0.55, 0.48, 0.51, 0.49]
    # f_stat, p_value = test_mean(values)
    #
    # print(f"F-statistic: {f_stat}")
    # print(f"P-value: {p_value}")
    #
    # if p_value < 0.05:
    #     print("The mean of the values is statistically different from 0.5.")
    # else:
    #     print("The mean of the values is not statistically different from 0.5.")
