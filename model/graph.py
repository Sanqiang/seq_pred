import tensorflow as tf
from model.arguments import get_args
from tensor2tensor.models import transformer
from tensor2tensor.layers import common_attention
from util import constant
from tensorflow.contrib.layers import xavier_initializer
from model.loss import sequence_loss


args = get_args()

class Graph:
    def __init__(self, is_train):
        self.hparams = transformer.transformer_base()
        self.is_train = is_train

    def get_embedding(self, ids):
        if not ids:
            return []
        else:
            return [tf.nn.embedding_lookup(self.emb, id) for id in ids]

    def create_model(self):
        # Input
        with tf.variable_scope('embedding'):
            self.emb = tf.get_variable(
                'embedding', [args.esize + constant.NUM_SPEC_MARK, args.dimension], tf.float32,
                initializer=xavier_initializer())

        with tf.variable_scope('inputs'):
            self.inputs_ph = []
            for step in range(args.max_simple_sentence):
                self.inputs_ph.append(
                    tf.zeros(args.batch_size, tf.int32, name='event'))

            self.inpt_events = tf.stack(
                [tf.zeros(args.batch_size, tf.int32, name='go')] + self.inputs_ph[:-1], axis=1)
            self.pred_events = tf.stack(
                self.inputs_ph, axis=1)

            encoder_attn_bias = common_attention.attention_bias_ignore_padding(
                tf.to_float(tf.equal(tf.stack(self.inpt_events, axis=1), constant.PAD_ID)))

        with tf.variable_scope('model'):
            outputs = transformer.transformer_encoder(self.inpt_events, encoder_attn_bias, self.hparams, 'trans')
            self.w = tf.get_variable(
                'output_w', [args.esize + constant.NUM_SPEC_MARK, args.dimension], tf.float32,
                initializer=xavier_initializer())
            self.b = tf.get_variable(
                'output_w', [args.esize + constant.NUM_SPEC_MARK], tf.float32,
                initializer=xavier_initializer())
            logits = tf.nn.xw_plus_b(outputs, tf.transpose(self.w), self.b)


        with tf.variable_scope('loss'):
            self.loss = sequence_loss(logits=logits,
                                      targets=self.pred_events)

        with tf.variable_scope('optim'):
            self.global_step = tf.get_variable(
                'global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

            if self.is_train:
                self.increment_global_step = tf.assign_add(self.global_step, 1)
                opt = tf.train.AdagradOptimizer(args.learning_rate)
                grads_and_vars = opt.compute_gradients(self.loss, var_list=tf.trainable_variables())
                grads = [g for (g, v) in grads_and_vars]
                clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
                self.train_op = opt.apply_gradients(
                    zip(clipped_grads, tf.trainable_variables()), global_step=self.global_step)

        print('Graph Built.')




