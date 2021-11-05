import logging
import os


# helper class to handle logging
class Logger:
    def __init__(self):
        logging.basicConfig(filename='./logs/output.log', level=logging.INFO)

    def log(self, message: str) -> None:
        logging.info(message)


# helper class to store results in the results file
class ResultFileWriter:
    result_file = None
    result_file_path = "./output/results.txt"

    def __init__(self):
        # remove existing results file if exists
        if os.path.exists(self.result_file_path):
            os.remove(self.result_file_path)

        # open results file in "append" mode
        self.result_file = open(self.result_file_path, 'a')

    def store_prediction_result(self, gene_id: str, localization: str):
        self.result_file.write(f"{gene_id},{localization}\n")

    def close(self):
        self.result_file.close()
