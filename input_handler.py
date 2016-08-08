import os
import collections
from six.moves import cPickle
import numpy as np

VOCABULARY_FILE = "vocabulary.pickle"


class InputHandler():
	"""
	This class handles loading the text from file, splitting it to batches, etc.
	"""
	def __init__(self, data_dir, batch_size, result_sequence_length):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.result_sequence_length = result_sequence_length

		input_file = os.path.join(data_dir, "input.txt")
		tensor_file = os.path.join(data_dir, "data.npy")
		vocabulary_file = os.path.join(data_dir, "%s" % VOCABULARY_FILE)

		print("reading text file")
		self.preprocess_data(input_file, vocabulary_file, tensor_file)

		self.create_batches()
		self.set_batch_pointer_to_zero()

	def preprocess_data(self, input_file, vocabulary_file, tensor_file):
		with open(input_file, "r") as f:
			data = f.read()

		x_text = data.split()

		self.vocabulary, self.words = self.build_vocabulary_and_inverse_vocabulary(x_text)
		self.vocabulary_size = len(self.words)

		# Save the words for when running the sampler
		with open(vocabulary_file, 'wb') as f:
			cPickle.dump(self.words, f)

		# index of words as our basic data
		self.tensor = np.array(list(map(self.vocabulary.get, x_text)))
		# Save the data to data.npy
		np.save(tensor_file, self.tensor)

	def build_vocabulary_and_inverse_vocabulary(self, words):
		"""
		Build the vocabulary and an inverse vocabulary that will be used to find
		the index of words in the vocabulary
		"""
		# count for each word how many times it appeared
		word_counts = collections.Counter(words)

		# Mapping from index to word
		inverse_vocabulary = [x[0] for x in word_counts.most_common()]
		inverse_vocabulary = list(sorted(inverse_vocabulary))

		# Mapping from word to index in the inverse vocabulary
		vocabulary = {x: i for i, x in enumerate(inverse_vocabulary)}
		return vocabulary, inverse_vocabulary

	def get_next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1
		return x, y

	def set_batch_pointer_to_zero(self):
		self.pointer = 0

	def create_batches(self):
		self.num_batches = int(self.tensor.size / (self.batch_size * self.result_sequence_length))
		if self.num_batches==0:
			assert False, "Not enough data. Make result_sequence_length and batch_size smaller."

		self.tensor = self.tensor[:self.num_batches * self.batch_size * self.result_sequence_length]
		xdata = self.tensor
		ydata = np.copy(self.tensor)

		ydata[:-1] = xdata[1:]
		ydata[-1] = xdata[0]
		self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
		self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
