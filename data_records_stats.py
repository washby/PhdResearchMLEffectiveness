from datetime import datetime
import logging
from stats_methods import DataCompiler

if __name__ == '__main__':
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=fr'logs\stats_{date_str}.log', filemode='w')

    dc = DataCompiler()

    # print(methods.stats_table_by_campus())
    print(dc.model_stats_table())

