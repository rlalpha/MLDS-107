import tensorflow as tf
import numpy as np
from base_model import BaseModel
import math


class Seq2SeqModel(BaseModel):
    def __init__(self, config):
        super(Seq2SeqModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config['max_to_keep'])

    def train(self, X_batch, y_inputs_batch, y_targets_batch, sequence_length_batch, fake_max_sequence, sample_prob_input):
        return self.sess.run([self.train_op, self.cost, self.pred_output],
                             feed_dict={self.encoder_inputs: X_batch, self.decoder_inputs: y_inputs_batch,
                                        self.decoder_targets: y_targets_batch, self.sequence_length: sequence_length_batch,
                                        self.sequence_length_fake: fake_max_sequence, self.sampling_prob: sample_prob_input,
                                        self.batch_size: self.config['BATCH_SIZE'], self.keep_prob: self.config['KEEP_PROB']})

    def build_model(self):

        NUM_OF_WORDS = self.config['NUM_OF_WORDS']
        MAX_LENGTH = self.config['MAX_LENGTH']
        ENCODER_INPUT_SIZE = self.config['ENCODER_INPUT_SIZE']
        EMBEDDING_SIZE = self.config['EMBEDDING_SIZE']
        NUM_OF_LAYER = self.config['NUM_OF_LAYER']
        HIDDEN_LAYER_SIZE = self.config['HIDDEN_LAYER_SIZE']
        BOS = self.config['BOS']
        EOS = self.config['EOS']
        USING_ATTENTION = self.config['USING_ATTENTION']

        # Define the model
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        with tf.name_scope('input'):
            # self.is_training = tf.placeholder(tf.bool)

            self.encoder_inputs = tf.placeholder(
                tf.float32, shape=[None, None, ENCODER_INPUT_SIZE])
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None])
            self.decoder_targets = tf.placeholder(tf.int32, shape=[None, None])
            self.sequence_length = tf.placeholder(tf.int32, shape=[None])
            self.sequence_length_fake = tf.placeholder(tf.int32, shape=[None])

            self.sampling_prob = tf.placeholder(tf.float32, shape=[])
            self.batch_size = tf.placeholder(tf.int32, shape=[])
            self.keep_prob = tf.placeholder(tf.float32, shape=[])

        # network architecture
        # Define Encoder
        with tf.name_scope('encoder'):
            encoder_inputs_embedded = tf.layers.dense(
                self.encoder_inputs, EMBEDDING_SIZE)
            encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(HIDDEN_LAYER_SIZE), self.keep_prob) for _ in range(NUM_OF_LAYER)])
            encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(HIDDEN_LAYER_SIZE), self.keep_prob) for _ in range(NUM_OF_LAYER)])
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw, encoder_cell_bw,
                                                                             encoder_inputs_embedded,
                                                                             dtype=tf.float32)
            encoder_outputs = tf.concat(encoder_outputs, 2)

        # Define Decoder for training
        with tf.name_scope('training_decoder'):
            decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(HIDDEN_LAYER_SIZE), self.keep_prob) for _ in range(NUM_OF_LAYER)])
            # embedding for decoder
            embeddings = tf.Variable(tf.random_uniform(
                [NUM_OF_WORDS, EMBEDDING_SIZE], -1.0, 1.0), dtype=tf.float32)
            decoder_inputs_embedded = tf.nn.embedding_lookup(
                embeddings, self.decoder_inputs)

            if USING_ATTENTION:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=HIDDEN_LAYER_SIZE, memory=encoder_outputs)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism, attention_layer_size=HIDDEN_LAYER_SIZE)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, NUM_OF_WORDS
                )
            else:
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, NUM_OF_WORDS)

            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_inputs_embedded,
                                                                                  self.sequence_length_fake, embeddings, self.sampling_prob)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell,
                                                               training_helper,
                                                               initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size))
            # unrolling the decoder layer
            training_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True)

        # Define Decoder for inference
        with tf.variable_scope('inference_decoder', reuse=True):
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                        tf.fill(
                                                                            [self.batch_size], BOS),
                                                                        EOS)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell,
                                                                inference_helper,
                                                                initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size))

            inference_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                        impute_finished=True, maximum_iterations=MAX_LENGTH)

        # Define the training logits
        training_logits = tf.identity(
            training_outputs.rnn_output, name='logits')
        self.pred_output = tf.identity(inference_outputs.rnn_output, name='logits')
        masks = tf.sequence_mask(self.sequence_length, MAX_LENGTH,
                                 name='mask', dtype=tf.float32)

        # Define training
        with tf.name_scope("optimization"):
            # Loss function - weighted softmax cross entropy
            self.cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                self.decoder_targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(1e-3)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)
                                for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)
            tf.summary.scalar('loss', self.cost)
