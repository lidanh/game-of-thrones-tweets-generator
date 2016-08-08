import tweepy
from text_cleaner import clean_text

import credentials
consumer_key = credentials.consumer_key
consumer_secret = credentials.consumer_secret
access_token = credentials.access_token
access_token_secret = credentials.access_token_secret

class MyStreamListener(tweepy.StreamListener):
	def __init__(self, outfile):
		self.outfile = outfile
		super( MyStreamListener, self ).__init__()

	def __enter__(self):
		"""
		define enter and exit so that the object can be used with 'with' statement
		"""
		self.file = open(self.outfile, "a")
		self.counter = 0
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		"""
		define enter and exit so that the object can be used with 'with' statement
		"""
		self._print_counter()
		self.file.close()

	def on_status(self, status):
		if self._is_retweet(status):
			return

		self.counter += 1
		if self.counter % 20 == 0:
			self._print_counter()

		self.file.write(clean_text(status.text))
		self.file.write('\n')


	def _is_retweet(self, status):
		"""
		Check if a tweet is a retweet or not
		"""
		return 'RT @' in status.text

	def _print_counter(self):
		print "Counter is {}".format(self.counter)

def main():
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)

	with MyStreamListener('game-of-thrones/input.txt') as myStreamListener:
		myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
		myStream.filter(track=['GameofThrones'], languages=["en"])


if __name__ == '__main__':
	main()
