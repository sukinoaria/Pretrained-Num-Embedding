import time,math,logging

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words.item()
        self.n_correct += stat.n_correct.item()

    def accuracy(self):
        return 100 * (float(self.n_correct) / self.n_words)

    def xent(self):
        return self.loss / self.n_words

    def ppl(self):
        return math.exp(min(self.loss / int(self.n_words), 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        logging.info(("Epoch %2d, %5d/%5d; loss: %6.2f; acc: %6.2f; ppl: %6.2f; xent: %6.2f; " +
                      "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                     (epoch, batch, n_batches, self.loss,
                      self.accuracy(),
                      self.ppl(),
                      self.xent(),
                      self.n_src_words / (t + 1e-5),
                      self.n_words / (t + 1e-5),
                      t))
    def output_dev(self, epoch):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        logging.info(("Dev:   Epoch %2d; loss: %6.2f; acc: %6.2f; ppl: %6.2f; xent: %6.2f; " +
                      "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                     (epoch, self.loss,
                      self.accuracy(),
                      self.ppl(),
                      self.xent(),
                      self.n_src_words / (t + 1e-5),
                      self.n_words / (t + 1e-5),
                      t))
