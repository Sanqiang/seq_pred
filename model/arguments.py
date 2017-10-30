import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Model Parameter')

    # Data
    parser.add_argument('-trpkl', '--train_data_pkl', default='../seq_pred_data/meme/train.pkl',
                        help='The Usage Model?')
    parser.add_argument('-vapkl', '--val_data_pkl', default='../seq_pred_data/meme/dev.pkl',
                        help='The Usage Model?')
    parser.add_argument('-tepkl', '--test_data_pkl', default='../seq_pred_data/meme/test.pkl',
                        help='The Usage Model?')
    parser.add_argument('-ldir', '--logdir', default='../seq_pred_fd/log/',
                        help='Dir for log?')

    # Graph
    parser.add_argument('-mlen', '--max_len', default=10, type=int,
                        help='Max length of sequence?')
    parser.add_argument('-esize', '--event_size', default=5000, type=int,
                        help='Size of event?')
    parser.add_argument('-dim', '--dimension', default=512, type=int,
                        help='Size of hidden state?')
    parser.add_argument('-bsize', '--batch_size', default=32, type=int,
                        help='Size of mini-batch?')

    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='Value of learning rate?')

    args = parser.parse_args()
    return args