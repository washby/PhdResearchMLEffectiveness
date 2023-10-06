import pyodbc
import json
import logging
import os

logger = logging.getLogger(__name__)


class DatabaseUtility:

    def __init__(self, filename):
        self.__config = None
        with open(filename) as f:
            self.__config = json.load(f)

        # Connect to the database
        self.__server = self.__config["server_name"]
        self.__database = self.__config['database']
        self.__username = self.__config['username']
        self.__table_name = self.__config['table_name']
        self.__cnxn = None
        self.__cursor = None
        self.__engine = None

    def establish_connection(self):
        password = input("Enter database password: ")
        os.system('cls')
        print("Enter database password: *****")
        driver = '{ODBC Driver 18 for SQL Server}'
        cnxn_str = f'Driver={driver};Server=tcp:{self.__server},1433;'
        cnxn_str += f'Database={self.__database};Uid={self.__username};Pwd={password};'
        cnxn_str += f'Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
        self.__cnxn = self.__cnxn = pyodbc.connect(cnxn_str)
        # Create a cursor object
        self.__cursor = self.__cnxn.cursor()

    def select_all_from_table(self, table_name):
        """
        Select all records from a table and return it as a pandas dataframe
        :param table_name: string of the table name
        :return: a pandas dataframe of the table
        """
        if self.__cnxn is None:
            logging.warning("Connection to database not established. Establishing connection now.")
            self.establish_connection()

        query_str = f'SELECT * FROM {table_name}'
        return self.__cursor.execute(query_str).fetchall()


    def close_connection(self):
        # Close the connection
        if self.__cnxn is not None:
            logger.info("Closing the database connection.")
            self.__cnxn.close()
        else:
            logger.info("No database connection is open. Therefore close() does nothing.")
