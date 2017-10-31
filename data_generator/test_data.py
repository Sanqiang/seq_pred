from util.path import get_path
import pickle as pkl
from model.arguments import get_args
import random as rd


args = get_args()


class TrainData:
    def __init__(self, mode='test'):
        self.pkl_data = pkl.load(
            open(get_path(args.train_data_pkl), 'rb'),
            encoding='latin1')
        self.data = self.pkl_data[mode]
        self.size = len(self.data)
        print('Load %s data!' % mode)

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        data_sample = self.data[i]
        return [map['type_event'] for map in data_sample]


if __name__ == '__main__':
    TrainData().get_data_sample()