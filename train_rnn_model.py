import tensorflow as tf
import argparse
import time
import os

from __future__ import print_function
from six.moves import cPickle

from input_handler import InputHandler
from rnnmodel import RNNModel

WORDS_VOCABULARY_FILE = 'words_vocabulary.pickle'

CONFIGURATION_FILE = 'configuration.pickle'

# Use on a computer with a GPU to perform a lot of computations
FAST_COMPUTER = True

def main():
	parser = argparse.ArgumentParser()
	args = parser.parse_args()

	args.data_dir = 'data/game-of-thrones'
	args.snapshots_dir = 'model_snapshots'

	args.training_rate = 0.005
	args.decay_rate = 0.95
	args.snapshot = 100
	args.gradient = 3

	if FAST_COMPUTER:
		args.rnn_size = 128
		args.network_depth = 2
		args.batch_size = 100
		args.result_length = 32
		args.num_epochs = 64
	else:
		args.rnn_size = 2
		args.network_depth = 2
		args.batch_size = 200
		args.result_length = 4
		args.num_epochs = 1

	train_model(args)

def train_model(args):
	data_loader = InputHandler(args.data_dir, args.batch_size, args.result_length)
	args.vocabulary_size = data_loader.vocabulary_size

	# Save the original files, so that we can load the model when sampling
	with open(os.path.join(args.snapshots_dir, CONFIGURATION_FILE), 'wb') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.snapshots_dir, WORDS_VOCABULARY_FILE), 'wb') as f:
		cPickle.dump((data_loader.words, data_loader.vocabulary), f)

	model = RNNModel(args.rnn_size, args.network_depth, args.batch_size, args.result_length,
					 args.vocabulary_size, args.gradient)

	with tf.Session() as session:
		tf.initialize_all_variables().run()
		saver = tf.train.Saver(tf.all_variables())
		for e in range(args.num_epochs):
			session.run(tf.assign(model.lr, args.training_rate * (args.decay_rate ** e)))
			data_loader.set_batch_pointer_to_zero()
			state = model.initial_state.eval()

			for b in range(data_loader.num_batches):
				x, y = data_loader.get_next_batch()
				feed = {model.input_data: x, model.targets: y, model.initial_state: state}
				train_loss, state, _ = session.run([model.cost, model.final_state, model.train_op], feed)
				if (e * data_loader.num_batches + b) % args.snapshot == 0 \
						or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
					snapshot_path = os.path.join(args.snapshots_dir, 'model.ckpt')
					saver.save(session, snapshot_path, global_step = e * data_loader.num_batches + b)
					print("Model snapshot was taken to {}".format(snapshot_path))

if __name__ == '__main__':
	main()
