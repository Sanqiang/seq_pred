from model.arguments import get_args
from data_generator.test_data import TestData
from util import constant
from model.graph import Graph
import tensorflow as tf
import numpy as np
from util.path import get_path
import os


args = get_args()
data = TestData()

def get_data(inputs_ph, it):
    input_feed = {}

    tmp_events = []
    last_events = []
    is_end = False
    effective_batch_size = 0
    for i in range(args.batch_size):
        if not is_end:
            events = next(it)
            effective_batch_size += 1
        if events is None or is_end:
            # End of dataset
            if not is_end:
                effective_batch_size -= 1
            is_end = True
            events = []

        events.insert(0, constant.START_ID)
        last_events.append(events[-1])
        if len(events) < args.max_len:
            num_pad = args.max_len - len(events)
            events.extend(num_pad * [constant.PAD_ID])
        else:
            events = events[:args.max_len]
        tmp_events.append(events)


    for step in range(args.max_len):
        input_feed[inputs_ph[step].name] = [tmp_events[batch_idx][step]
                                            for batch_idx in range(args.batch_size)]

    return input_feed, last_events, effective_batch_size, is_end


def test():
    it = data.get_data_iter()
    graph = Graph(is_train=False)
    tf.reset_default_graph()
    graph.create_model()

    sv = tf.train.Supervisor(logdir=get_path(args.logdir),
                             global_step=graph.global_step)
    sess = sv.PrepareSession()
    losses = []
    scores = 0
    total = 0
    while True:
        input_feed, gt_last_events, effective_batch_size, is_end = get_data(graph.inputs_ph, it)
        fetches = [graph.last_event, graph.loss, graph.global_step]
        last_events, loss, step = sess.run(fetches, input_feed)

        for i in range(effective_batch_size):
            losses.append(loss)

            total += 1
            if last_events[i] == gt_last_events[i]:
                scores += 1

        if is_end:
            break

    file = 'Accurary%sLoss%s.txt' % (scores/total, np.mean(losses))
    output = 'Accurary:\t%s Loss:\t%s' % (scores/total, np.mean(losses))
    if not os.path.exists(get_path(args.resultdir)):
        os.mkdir(get_path(args.resultdir))
    f = open(get_path(args.resultdir) + file, mode='w', encoding='utf-8')
    f.write(output)
    f.close()


if __name__ == '__main__':
    test()