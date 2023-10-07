import json
import logging
from utils import create_pickle_file

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='create_pickles.log', filemode='w')
    with open('data_config.json', 'r') as f:
        data_config = json.load(f)
    for campus in data_config:
        print(f"Creating pickle file for {campus['name']} from {campus['table_name']}")
        logging.info("Creating pickle file for {} from {}".format(campus['name'], campus['table_name']))
        create_pickle_file(campus['table_name'], f'{campus["name"]}.pkl')