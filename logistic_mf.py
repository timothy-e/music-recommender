from collaborative import get_collab_matrix
from utils import timeit, convert_to_rank
from scipy.sparse import coo_matrix
from typing import Tuple, Optional
import sys
import numpy as np


class LogisticMF:
    def __init__(
        self,
        M: coo_matrix,
        n_latent_factors: int,
        alpha: float,
        l2_regulation: float,
        gamma: float,
        iterations: int,
        partition_size: Optional[Tuple[int, int]] = None
    ):
        self.ones = np.ones(shape=M.shape)

        self.sparseM = M
        self.M = M.todense() * alpha + self.ones  # n x m
        self.n_users, self.n_songs = M.shape

        self.n_latent_factors = n_latent_factors
        self.alpha = alpha
        self.l2_regulation = l2_regulation
        self.gamma = gamma
        self.iterations = iterations
        self.partition_size = partition_size

    @timeit(bold=True)
    def train(self):
        """
        Calculate and store the biases and vectors
        """
        self.user_vecs = np.random.normal(
            size=(self.n_users, self.n_latent_factors))  # n x f
        self.song_vecs = np.random.normal(
            size=(self.n_songs, self.n_latent_factors))  # m x f

        self.user_biases = np.random.normal(size=(self.n_users, 1))  # n x 1
        self.song_biases = np.random.normal(size=(self.n_songs, 1))  # m x 1

        ddx_user_vec_sum = np.zeros(
            shape=(self.n_users, self.n_latent_factors))  # n x f
        ddx_song_vec_sum = np.zeros(
            shape=(self.n_songs, self.n_latent_factors))  # m x f
        ddx_user_bias_sum = np.zeros(shape=(self.n_users, 1))  # n x 1
        ddx_song_bias_sum = np.zeros(shape=(self.n_songs, 1))  # m x 1

        @timeit(dark=True)
        def train_iteration(
                vecs, biases, ddx_vec, ddx_bias, ddx_vec_sum, ddx_bias_sum):
            """Modify vecs, biases, ddx_vec_sum, and ddx_bias_sum with
            gradient descent"""
            # let z be one of [n, m]
            ddx_vec_sum += np.square(ddx_vec)  # z x f
            ddx_bias_sum += np.square(ddx_bias)  # z x 1

            vec_step_size = self.gamma / np.sqrt(ddx_vec_sum)  # z x f
            bias_step_size = self.gamma / np.sqrt(ddx_bias_sum)  # z x 1

            vecs += np.multiply(vec_step_size, ddx_vec)  # z x f   *   z x f
            biases += np.multiply(bias_step_size, ddx_bias)  # z x 1  *  z x 1

        for i in range(self.iterations):
            ddx_user_vec, ddx_user_bias = self.user_derivative(
                self.M, self.user_vecs, self.song_vecs,
                self.user_biases, self.song_biases, self.ones
            )
            train_iteration(
                vecs=self.user_vecs,
                biases=self.user_biases,
                ddx_vec=ddx_user_vec,
                ddx_bias=ddx_user_bias,
                ddx_vec_sum=ddx_user_vec_sum,
                ddx_bias_sum=ddx_user_bias_sum,
            )

            ddx_song_vec, ddx_song_bias = self.song_derivative(
                self.M, self.user_vecs, self.song_vecs,
                self.user_biases, self.song_biases, self.ones
            )
            train_iteration(
                vecs=self.song_vecs,
                biases=self.song_biases,
                ddx_vec=ddx_song_vec,
                ddx_bias=ddx_song_bias,
                ddx_vec_sum=ddx_song_vec_sum,
                ddx_bias_sum=ddx_song_bias_sum,
            )

    def user_derivative(
        self, M, user_vecs, song_vecs, user_biases, song_biases, ones
    ):
        # n x m @ m x f = n x f
        ddx_vec = M @ song_vecs
        ddx_bias = np.sum(M, axis=1)  # n x 1

        A = self._common_derivative(
            M, user_vecs, song_vecs, user_biases, song_biases, ones
        )  # n x m

        ddx_vec -= A @ song_vecs  # n x m @ m x f = n x f
        ddx_vec -= self.l2_regulation * user_vecs  # n x f

        ddx_bias -= np.expand_dims(np.sum(A, axis=1), 1)  # n x 1

        return (ddx_vec, ddx_bias)

    def song_derivative(
        self, M, user_vecs, song_vecs, user_biases, song_biases, ones
    ):
        ddx_vec = M.T @ user_vecs
        ddx_bias = np.sum(M, axis=0).T

        A = self._common_derivative(
            M, user_vecs, song_vecs, user_biases, song_biases, ones
        )

        ddx_vec -= A.T @ user_vecs
        ddx_vec -= self.l2_regulation * song_vecs
        ddx_bias -= np.expand_dims(np.sum(A, axis=0), 1)

        return (ddx_vec, ddx_bias)

    def _common_derivative(
        self, M, user_vecs, song_vecs, user_biases, song_biases, ones
    ):
        """Return e^a/(1+e^a) for every a in a matrix created by a
        combination of user and song vectors and biases"""
        A = user_vecs @ song_vecs.T  # n x f @ f x m = n x m
        A += user_biases
        A += song_biases.T
        A = np.exp(A)
        A /= A + ones
        A *= M + ones
        return A

    @timeit()
    def log_likelihood(self):
        """Return a single number of how well this model performs"""

        likelihood = 0
        A = self.user_vecs @ self.song_vecs.T
        A += self.user_biases + self.song_biases.T
        B = np.multiply(A, self.M)
        likelihood += np.sum(B)

        del B

        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = np.multiply(self.M + self.ones, A)
        likelihood -= np.sum(A)

        del A

        likelihood -= 0.5 * self.l2_regulation * \
            np.sum(np.square(self.user_vecs))
        likelihood -= 0.5 * self.l2_regulation * \
            np.sum(np.square(self.song_vecs))

        return likelihood

    @timeit(bold=True)
    def get_rank_matrix(self) -> np.ndarray:
        """Return a n*m matrix of ranks, where each position from M has the
        value -1"""
        newM = self.user_vecs @ self.song_vecs.T
        for r, c in zip(self.sparseM.row, self.sparseM.col):
            newM[r, c] = -1

        return convert_to_rank(newM)

    @timeit()
    def write_vectors(self, fp_users: str, fp_songs: str):
        """Output user vectors to file `fp_users` and item vectors to file
        `fp_songs`"""

        def write_array(f, arr):
            for vec in arr:
                f.write(f"{','.join(str(a) for a in vec)}\n")

        with open(fp_users, "w+") as f:
            write_array(f, self.user_vecs)

        with open(fp_songs, "w+") as f:
            write_array(f, self.song_vecs)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    scale = -arg.count("s")
    user_labels, track_labels, M = get_collab_matrix(
        scale=10 ** scale, fp="mini_triplets.csv"
    )

    lmf = LogisticMF(M, n_latent_factors=5, alpha=2,
                     l2_regulation=1, gamma=0.5, iterations=5)
    lmf.train()
    print(lmf.log_likelihood())
    print(lmf.get_rank_matrix())
