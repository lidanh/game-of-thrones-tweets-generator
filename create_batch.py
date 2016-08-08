from __future__ import print_function

from random import randint
import os

import tensorflow as tf
from six.moves import cPickle

from rnnmodel import RNNModel

VOCABULARY_FILE = 'words_vocabulary.pickle'

CONFIGURATION_FILE = 'configuration.pickle'


def main():

	# typical tweet length
	number_of_words = randint(8,12)

	create_batch(model_snapshots_dir='model_snapshots', number_of_words=number_of_words)


def create_batch(model_snapshots_dir, number_of_words):

	# Load original configuration
	with open(os.path.join(model_snapshots_dir, CONFIGURATION_FILE), 'rb') as configuration_file:
		saved_args = cPickle.load(configuration_file)

	# For sampling use batch and sequence size of 1
	saved_args.batch_size = 1
	saved_args.seq_length = 1

	# Init with the original arguments that were given to the model
	model = RNNModel(saved_args.rnn_size, saved_args.num_layers, saved_args.batch_size,saved_args.seq_length,
					 saved_args.vocabulary_size, saved_args.gradient_clip, sample=True)

	# Load the words and vocabulary created originally
	with open(os.path.join(model_snapshots_dir, VOCABULARY_FILE), 'rb') as vocabulary_file:
		words, vocabulary = cPickle.load(vocabulary_file)


	with tf.Session() as session:

		# Init tensorflow
		tf.initialize_all_variables().run()

		# Load saved data and generate the sample
		saver = tf.train.Saver(tf.all_variables())

		# Make sure that the model is saved as expected
		checkpoint = tf.train.get_checkpoint_state(model_snapshots_dir)

		# load the tensorflow data to a session, and run the model on it
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(session, checkpoint.model_checkpoint_path) # restore the LSTM
			sample = model.sample(session, words, vocabulary, number_of_words)
			print(sample)
		else:
			print('Could not load the saved data.\n'
				  'Did you run train_rnn_model.py?')


if __name__ == '__main__':
	main()
