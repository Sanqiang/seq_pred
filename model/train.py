from model.arguments import get_args
from data_generator.train_data import TrainData
from util import constant
from model.graph import Graph
import tensorflow as tf
import numpy as np

args = get_args()
data = TrainData()


def get_data(inputs_ph):
    input_feed = {}

    tmp_events = []
    for i in range(args.batch_size):
        events = data.get_data_sample()

        if len(events) < args.max_len:
            num_pad = args.max_len - len(events)
            events.extend(num_pad * constant.PAD_ID)
        else:
            events = events[:args.max_len]

        tmp_events.append(events)

    for step in range(args.max_len):
        input_feed[tmp_events[step].name] = [tmp_events[batch_idx][step]
                                             for batch_idx in range(args.batch_size)]

    return input_feed


def train():
    graph = Graph()
    graph.create_model()


    sv = tf.train.Supervisor(logdir=args.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver,
                             save_model_secs=600)
    sess = sv.PrepareSession(config=tf.ConfigProto())
    losses = []
    while True:
        input_feed = get_data(graph.inputs_ph)
        fetches = [graph.train_op, graph.loss, graph.global_step]
        _, loss, step = sess.run(fetches, input_feed)
        losses.append(loss)

        if step % 1000 == 0:
            print('Loss\t%s.' % np.mean(losses))

