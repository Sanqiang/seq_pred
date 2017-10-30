import os


def get_path(file_path):
    return os.path.dirname(os.path.abspath(__file__)) + '/../' + file_path