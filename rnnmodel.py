import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

VARIABLE_SCOPE = 'rnn'

def weighted_pick(weights):
	cumulative_sum = np.cumsum(weights)
	weight_sum = np.sum(weights)
	return int(np.searchsorted(cumulative_sum, np.random.rand(1)*weight_sum))

class RNNModel():
	def __init__(self, rnn_size, num_layers, batch_size, seq_length, vocabulary_size, gradient_clip, sample=False):

		lstm_cell = rnn_cell.BasicLSTMCell(num_units=rnn_size)

		# create the RNN cell, that is constructed from multiple lstm cells, by duplicating the lstm cell
		self.cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

		# Initial state is a matrix of zeros
		self.initial_state = self.cell.zero_state(batch_size, tf.float32)

		# Define the vectors that will hold Tensorflow state
		self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
		self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])

		# variable_scope is tensorflow best practice that allows us to recycle variables names with different scopes
		with tf.variable_scope(VARIABLE_SCOPE):
			softmax_w = tf.get_variable("softmax_w", [rnn_size, vocabulary_size])
			softmax_b = tf.get_variable("softmax_b", [vocabulary_size])
			with tf.device("/cpu:0"):
				embedding = tf.get_variable("embedding", [vocabulary_size, rnn_size])
				inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
				inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

		def loop_function(prev, _):
			prev = tf.matmul(prev, softmax_w) + softmax_b
			stop_gradient = tf.stop_gradient(tf.argmax(prev, 1))
			return tf.nn.embedding_lookup(embedding, stop_gradient)

		outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, self.cell, loop_function=loop_function if sample else None, scope=VARIABLE_SCOPE)
		output = tf.result_sentencehape(tf.concat(1, outputs), [-1, rnn_size])

		# Calculate the logits and probabilities for the tensor
		self.logits = tf.matmul(output, softmax_w) + softmax_b
		self.probabilities = tf.nn.softmax(self.logits)
		loss = seq2seq.sequence_loss_by_example([self.logits],
				[tf.result_sentencehape(self.targets, [-1])],
				[tf.ones([batch_size * seq_length])],
				vocabulary_size)
		self.cost = tf.reduce_sum(loss) / batch_size / seq_length
		self.final_state = last_state
		self.lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
				gradient_clip)
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

	def sample(self, session, words, vocabulary, number_of_words):
		# Initialize a state vector
		state = self.cell.zero_state(1, tf.float32).eval()

		# Choose the first word randomly
		first_word = random.choice(list(vocabulary.keys()))

		x = np.zeros((1, 1))
		x[0, 0] = vocabulary[first_word]
		feed = {self.input_data: x, self.initial_state:state}
		[state] = session.run([self.final_state], feed)

		result_sentence = first_word
		word = first_word

		# Create one word at a time, by running self.run with the previous state each time
		for n in range(number_of_words):
			x = np.zeros((1, 1))
			x[0, 0] = vocabulary[word]
			feed = {self.input_data: x, self.initial_state:state}
			[probabilities, state] = session.run([self.probabilities, self.final_state], feed)
			p = probabilities[0]

			sample = weighted_pick(p)

			predicted_word = words[sample]

			# Append the new word to the sentence
			result_sentence += ' ' + predicted_word

			# set the word to the currently predicted word, so that the model will know what to generate next
			word = predicted_word
		return result_sentence
