import tensorflow as tf
from model.arguments import get_args
from tensor2tensor.models import transformer
from util import constant
from tensorflow.contrib.layers import xavier_initializer
from model.loss import sequence_loss
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers


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
                'embedding', [args.event_size + constant.NUM_SPEC_MARK, args.dimension], tf.float32,
                initializer=xavier_initializer())

        with tf.variable_scope('inputs'):
            self.inputs_ph = []
            for step in range(args.max_len):
                self.inputs_ph.append(
                    tf.zeros(args.batch_size, tf.int32, name='event'))

            self.inpt_events = [tf.zeros(args.batch_size, tf.int32, name='go')] + self.inputs_ph[:-1]
            self.inpt_events_emb = tf.stack(self.get_embedding(self.inpt_events), axis=1)

            if not self.is_train:
                self.pred_events = self.inputs_ph

            self_attention_bias = (
                common_attention.attention_bias_lower_triangle(args.max_len))

        with tf.variable_scope('model'):
            outputs = self.attention_lm_decoder(self.inpt_events_emb, self_attention_bias, self.hparams, 'trans')
            self.w = tf.get_variable(
                'output_w', [args.dimension, args.event_size + constant.NUM_SPEC_MARK], tf.float32,
                initializer=xavier_initializer())
            self.b = tf.get_variable(
                'output_b', [args.event_size + constant.NUM_SPEC_MARK], tf.float32,
                initializer=xavier_initializer())
            # logits = tf.nn.xw_plus_b(outputs, tf.transpose(self.w), self.b)
            logits = tf.nn.conv1d(outputs, tf.expand_dims(self.w, 0), 1, 'SAME')

        with tf.variable_scope('loss'):
            self.loss = sequence_loss(logits=logits,
                                      targets=tf.stack(self.pred_events, axis=1))

        with tf.variable_scope('optim'):
            self.global_step = tf.get_variable(
                'global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)

            if self.is_train:
                self.increment_global_step = tf.assign_add(self.global_step, 1)
                opt = tf.train.AdagradOptimizer(args.learning_rate)
                grads_and_vars = opt.compute_gradients(self.loss, var_list=tf.trainable_variables())
                grads = [g for (g, v) in grads_and_vars]
                clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
                self.train_op = opt.apply_gradients(
                    zip(clipped_grads, tf.trainable_variables()), global_step=self.global_step)
            else:
                self.last_event = tf.argmax(logits[:, -1, :], axis=-1)

            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        print('Graph Built.')

    def attention_lm_decoder(self,
                             decoder_input,
                             decoder_self_attention_bias,
                             hparams,
                             name="decoder"):
        """A stack of attention_lm layers.
           Coped from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/attention_lm.py
        Args:
          decoder_input: a Tensor
          decoder_self_attention_bias: bias Tensor for self-attention
            (see common_attention.attention_bias())
          hparams: hyperparameters for model
          name: a string
        Returns:
          y: a Tensors
        """
        x = decoder_input
        with tf.variable_scope(name):
            for layer in range(hparams.num_hidden_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            common_layers.layer_preprocess(x, hparams),
                            None,
                            decoder_self_attention_bias,
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout)
                        x = common_layers.layer_postprocess(x, y, hparams)
                    with tf.variable_scope("ffn"):
                        y = common_layers.conv_hidden_relu(
                            common_layers.layer_preprocess(x, hparams),
                            hparams.filter_size,
                            hparams.hidden_size,
                            dropout=hparams.relu_dropout)
                        x = common_layers.layer_postprocess(x, y, hparams)
            return common_layers.layer_preprocess(x, hparams)




