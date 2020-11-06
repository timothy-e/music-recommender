def open_matrix(fp):
    """
    Return the user-item matrix from file `fp`
    """
    pass


def log_likelihood(user_vecs, item_vecs, user_biases, item_biases):
    """
    Also should take in counts and ones??
    """
    pass


class LogisticMF:
    def __init__(self, iterations, **kwargs):
        self.iterations = iterations

    def train(self):
        """
        Calculate and store the biases and vectors
        """
        pass

    def deriv(self, user):
        """
        ???
        """

    def write_vectors(self):
        """
        Output user and item vectors to a file
        """
        pass