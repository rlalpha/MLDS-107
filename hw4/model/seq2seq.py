import tensorflow as tf
class Seq2seq(object):

    def __init__(self, sess, encoder_input_size, hidden_unit_size,
     input_embedding_size, vocab_size):

        # input set-up
        self.sess = sess
        self.encoder_inputs = tf.placeholder(tf.float32, shape = [None, None, encoder_input_size])
        self.decoder_inputs = tf.placeholder(tf.int32, shape = [None, None])
        self.decoder_targets = tf.placeholder(tf.int32, shape = [None, None])
        self.sequence_length = tf.placeholder(tf.int32, shape = [None])

        # embedding for decoder
        embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)

        # Define encoder
        encoder_cell = tf.contrib.rnn.LSTMCell(hidden_unit_size)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
            encoder_cell, self.encoder_inputs,
            dtype=tf.float32, time_major=False,
        )

        del encoder_outputs

        # Define Decoder
        decoder_cell = tf.contrib.rnn.LSTMCell(hidden_unit_size)

        decoder_outputs, _ = tf.nn.dynamic_rnn(
            decoder_cell, decoder_inputs_embedded,

            initial_state=encoder_final_state,

            dtype=tf.float32, time_major=False, scope="plain_decoder",
            
            sequence_length = self.sequence_length
        )

        decoder_logits = tf.layers.dense(decoder_outputs, vocab_size)

        self.decoder_prediction = tf.argmax(decoder_logits, 2)

        #Loss
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32),
            logits=decoder_logits,
        )

        self.loss = tf.reduce_mean(stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    
    def trian(self, x, y_inputs, y_targets, sequence_length):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict = {self.encoder_inputs : x,
        self.decoder_inputs : y_inputs, self.decoder_targets : y_targets, self.sequence_length : sequence_length})     
        print('loss : {0}'.format(loss))






    
    

     