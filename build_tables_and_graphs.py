from datetime import datetime
import logging
from stats_methods import DataCompiler
from os.path import join
from BuildExcelFile import build_excel_file


def build_campus_counts_table():
    results = dc.counts_table_by_campus()
    print(results)
    with open(join(stats_dir, 'stats_by_campus.txt'), 'w') as f:
        f.write(results)
    results = dc.fail_percents_table_by_campus()
    print(results)
    with open(join(stats_dir, 'fail_rate_by_campus.txt'), 'w') as f:
        f.write(results)


def build_measurements_tables():
    measurements = ['f1_measure', 'accuracy', 'mcc']
    for measurement in measurements:
        results = dc.measurement_stats_table(measurement)
        print(results)
        with open(join(stats_dir, f'{measurement}_stats.txt'), 'w') as f:
            f.write(results)

def build_histograms():
    measurements = ['f1_measure']
    for measurement in measurements:
        dc.create_histogram(measurement, title='F1 Values', save=True)


if __name__ == '__main__':
    stats_dir = 'stats_files'
    # date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #                     filename=fr'logs\stats_{date_str}.log', filemode='w')

    dc = DataCompiler()

    # build_campus_counts_table()
    # build_measurements_tables()

    # build_excel_file()
    # build_histograms()
    # dc.run_anova_test_info('f1_measure')

    dc.one_sample_t_test('f1_measure')