from util.path import get_path
import pickle as pkl
from model.arguments import get_args
import random as rd


args = get_args()


class TestData:
    def __init__(self, mode='test'):
        self.pkl_data = pkl.load(
            open(get_path(args.test_data_pkl), 'rb'),
            encoding='latin1')
        self.data = self.pkl_data[mode]
        self.size = len(self.data)
        print('Load %s data!' % mode)

    def get_data_iter(self):
        i = 0
        while True:
            yield [map['type_event'] for map in self.data[i]]

            i += 1
            if i == self.size:
                yield None


if __name__ == '__main__':
    TestData().get_data_iter()