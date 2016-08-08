# Game of Thrones - Tweets Generator

![Game of Thrones](http://www.pngall.com/wp-content/uploads/2016/05/Game-of-Thrones-Logo-PNG-Picture.png)

![Tensor Flow](https://www.tensorflow.org/images/logo-alt@2x.png)

## Introduction
This project is a tweets generator, which generates tweets about Game of Thrones using Tensor Flow and RNN models.

The model is trained by thousands of tweets collected in the last month about game of thrones, and build a RNN model (specifically *LSTM*) which produces new tweets based on the input data.

This project is the final project in data scientist workshop course, Ben Gurion University of the Negev, August 2016.

We learned how to use Google Tensorflow in [Google deep learning](https://www.udacity.com/course/deep-learning--ud730) course and built some self-learning projects before using this library. Therefore, the code of this project might be a bit complicated compared to what we've learned in the class.

## Data Scraping
We collect the tweets using `tweepy` python library.
The results are filtered for english only tweets because other languages might make the
model inconssistent.  

The script in which we have used to scrape the data can be found under [`data/download_got_tweets.py`](data/download_got_tweets.py).

Note that you must add your twitter developer credentials to [`credentials_template.py`](data/credentials_template.py), and rename it to `credentials.py` in order to download new data.

## Data Description
As mentioned, we chose to analyze tweets about Game of Thrones.
One of the challenges about this dataset is that it is not classic data for RNNs, because thousands of tweets are considered by the model to be a single long data item, when actually each tweet stands on its own. We were not sure if the RNN will be able to handle this challenge, yet, we have decided to try it.

This project is interesting mostly for entertainment and amusement reasons. It might also have a (semi-)commercial potential for generating tweets for content websites etc.

## Pre Processing
We use mostly raw data. We filtered out retweets and comments, as they are usually less relevant, but we use the whole tweet text. We also removed URLs, newlines, non-ascii chars, etc.

You can find the code of this stage in [`data/text-cleaner.py`](data/text-cleaner.py).

## Comparing the Results to the Real Data
The [`compare_results.py`](compare_results.py) can be executed on input and output files.

For each line of the output file, the script finds the line that is the most similar to it in the input,
and calculates a similarity score.
The score is based on testing for each word in the input, if it appears in the output and vice versa.
In addition, we compare the lengths of the sample and the actual tweet.
Then we add the length diff and words diff, with a weight function, to calculate the final score.

We Chose this way to compare the data, because many times, when generating/processing language, the order of the words in a sentence is less meaningful then the similarity of the words.

The average score that was produced is **0.354810741849** from range \[0,1\] (lower is better), you can run
the `compare_results.py` script on your own:

```sh
$ python compare_results.py --input_file data/game-of-thrones/input.txt --output_file data/game-of-thrones/output.txt
```

## LSTM
LSTM stands for **Long short-term memory**.
It is a recurrent neural network (RNN) architecture.  Unlike traditional RNNs, an LSTM network is well-suited to learn from experience to classify, process and predict time series when there are very long time lags of unknown size between important events. This is one of the main reasons why LSTM outperforms alternative RNNs and Hidden Markov Models and other sequence learning methods in numerous applications.

It is widely used for generating a stream of words. Each time when we look at word, we consider the probability for the next word, and choose the next word accordingly, repeatedly.

So, for example, if we have the word "Hello", and we known that after it the word "World" has probability
of 0.3, the word "Kitty" has probability of 0.5, and the word "Adele" has a probability of 0.2, then for 3/10 cases we will produce the wold "World" after writing "Hello", in 5/10 of the cases we will write "Kitty" and in 2/10 of the cases we will write "Adele".

You can find more information about LSTM [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and also in [deeplearning4j](http://deeplearning4j.org/lstm.html).

## Input and Output Files
The input file is located under [data/game-of-thrones/input.txt](data/game-of-thrones/input.txt). Each line represents a tweet.

The output file is located under [data/game-of-thrones/output.txt](data/game-of-thrones/output.txt).

## GPU Optimization
We have had a relatively large dataset (70,000+ tweets, ~800,000 words). Running the algorithm without
GPU-optimization takes more than 4 days.

We have used Tensorflow to do the calculations on the GPU, and so, with an `g2.2xlarge` EC2 machine. It took ~2 hours to run the algorithm and build the tweets model.

In order accelerate the training stage (larger RNN size, longer sequences, etc.) if GPU optimization is enabled, change `FAST_COMPUTER` to `True` under [train_rnn_model.py](train\_rnn_model.py) (default is `True`).


## Usage
To train the model, run:

```sh
$ python train_rnn_model.py
```

To get a sample batch, run:

```sh
$ python create_batch.py
```

## Train the Model in AWS
1. Follow [this guide](http://ramhiser.com/2016/01/05/installing-tensorflow-on-an-aws-ec2-instance-with-gpu-support/) in order to start an EC2 machine with GPU support. As mentioned earlier, it's better to train the model on `g2.2xlarge` instance, which is optimized for massive use of the GPU (and thus it's great in this use case).
2. Clone this repo in the machine, and run:


```sh
$ python train_rnn_model.py
```

  > Consider running in tmux to prevent network problems from terminating your training session.

3. Run `python create_batch.py` to get a sample of the data.

## References and Relevant Links
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Tensorflow Recurrent Neural Networks tutorial](https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html)
- [Tensorflow seq2seq tutorial](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html)
- [Generating Text with Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)

## LICENSE
The MIT License.

See [LICENSE](LICENSE.md)
