from collaborative import get_collab_matrix
from utils import timeit, convert_row_to_rank, matrix_mult, print_progress_bar
from scipy.sparse import coo_matrix, csr_matrix
from typing import Tuple, Optional, List
import sys
import numpy as np
import math


class LogisticMF:
    def __init__(
        self,
        M: coo_matrix,
        n_latent_factors: int,
        alpha: float,
        l2_regularization: float,
        gamma: float,
        iterations: int,
    ):
        self.coo_M: coo_matrix = M
        self.csr_M: csr_matrix = M.tocsr()
        self.n_users, self.n_songs = M.shape

        self.n_latent_factors = n_latent_factors
        self.alpha = alpha
        self.l2_regularization = l2_regularization
        self.gamma = gamma
        self.iterations = iterations

    @timeit(bold=True)
    def train(
        self, partition_size: Optional[Tuple[int, int]] = None, debug=False
    ):
        """
        Calculate and store the biases and vectors
        May calculate gradients in batches of size `partition_size`
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

        partition_size = partition_size or self.M.shape

        @timeit(dark=True)
        def subset_iteration(vecs, biases, ddx_vec_sum, ddx_bias_sum, ddx_fn):
            """Modify vecs, biases, ddx_vec_sum, and ddx_bias_sum with
            gradient descent"""
            ddx_vec = np.empty(shape=vecs.shape)
            ddx_bias = np.empty(shape=biases.shape)

            for u_start in range(0, self.n_users, partition_size[0]):
                for s_start in range(0, self.n_songs, partition_size[1]):
                    u_end = u_start + partition_size[0]
                    s_end = s_start + partition_size[1]
                    if debug:
                        print(
                            "Working on the subset from " +
                            f"({u_start}, {s_start}) to ({u_end}, {s_end})"
                        )

                    subset_csr_M = self.csr_M[u_start:u_end, s_start:s_end]
                    subset_ones = np.ones(shape=subset_csr_M.shape)
                    subset_M = subset_ones + self.alpha * subset_csr_M

                    subset_user_vecs = self.user_vecs[u_start:u_end, :]
                    subset_song_vecs = self.song_vecs[s_start:s_end, :]

                    subset_user_biases = self.user_biases[u_start:u_end, :]
                    subset_song_biases = self.song_biases[s_start:s_end, :]

                    subset_ddx_vec, subset_ddx_bias = ddx_fn(
                        subset_M, subset_user_vecs, subset_song_vecs,
                        subset_user_biases, subset_song_biases, subset_ones
                    )

                    # sad that I can't make this fn user/song agnostic
                    if vecs is self.user_vecs:
                        ddx_vec[u_start:u_end] = subset_ddx_vec
                        ddx_bias[u_start:u_end] = subset_ddx_bias
                    else:
                        ddx_vec[s_start:s_end] = subset_ddx_vec
                        ddx_bias[s_start:s_end] = subset_ddx_bias

            # let z be one of [n, m]
            ddx_vec_sum += np.square(ddx_vec)  # z x f
            ddx_bias_sum += np.square(ddx_bias)  # z x 1

            vec_step_size = self.gamma / np.sqrt(ddx_vec_sum)  # z x f
            bias_step_size = self.gamma / np.sqrt(ddx_bias_sum)  # z x 1

            vecs += np.multiply(vec_step_size, ddx_vec)  # z x f   *   z x f
            biases += np.multiply(bias_step_size, ddx_bias)  # z x 1  *  z x 1

        for i in range(self.iterations):
            subset_iteration(
                vecs=self.user_vecs,
                biases=self.user_biases,
                ddx_vec_sum=ddx_user_vec_sum,
                ddx_bias_sum=ddx_user_bias_sum,
                ddx_fn=self.user_derivative
            )

            subset_iteration(
                vecs=self.song_vecs,
                biases=self.song_biases,
                ddx_vec_sum=ddx_song_vec_sum,
                ddx_bias_sum=ddx_song_bias_sum,
                ddx_fn=self.song_derivative
            )

    def user_derivative(
        self, M, user_vecs, song_vecs, user_biases, song_biases, ones
    ):
        """Return derivative of user vector and bias"""
        # n x m @ m x f = n x f
        ddx_vec = M @ song_vecs
        ddx_bias = np.sum(M, axis=1)  # n x 1

        A = self._common_derivative(
            M, user_vecs, song_vecs, user_biases, song_biases, ones
        )  # n x m

        ddx_vec -= A @ song_vecs  # n x m @ m x f = n x f
        ddx_vec -= self.l2_regularization * user_vecs  # n x f

        ddx_bias -= np.expand_dims(np.sum(A, axis=1), 1)  # n x 1

        return (ddx_vec, ddx_bias)

    def song_derivative(
        self, M, user_vecs, song_vecs, user_biases, song_biases, ones
    ):
        """Return derivative of song vector and bias"""
        ddx_vec = M.T @ user_vecs
        ddx_bias = np.sum(M, axis=0).T

        A = self._common_derivative(
            M, user_vecs, song_vecs, user_biases, song_biases, ones
        )

        ddx_vec -= A.T @ user_vecs
        ddx_vec -= self.l2_regularization * song_vecs
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

    @timeit(bold=True)
    def log_likelihood(self):
        """Return a single number of how well this model performs"""
        sparseM = self.csr_M.todok()

        likelihood = 0
        for u, (user_vec, user_bias) in enumerate(zip(
            self.user_vecs, self.user_biases.flatten()
        )):
            print_progress_bar(u, self.n_users - 1,
                               prefix='Computing log likelihood:')
            xuys = self.song_vecs @ user_vec
            for s, (song_vec, song_bias) in enumerate(zip(
                self.song_vecs, self.song_biases.flatten()
            )):
                alpha_rus = self.alpha * \
                    sparseM[(u, s)] if (u, s) in sparseM else 0
                xu_ys_plus_biases = xuys[s] + user_bias + song_bias
                likelihood += alpha_rus * xu_ys_plus_biases - \
                    (1 + alpha_rus) * math.log(1 + math.exp(xu_ys_plus_biases))

        likelihood -= 0.5 * self.l2_regularization * \
            np.sum(np.square(self.user_vecs))
        likelihood -= 0.5 * self.l2_regularization * \
            np.sum(np.square(self.song_vecs))

        return likelihood

    def get_rank_matrix(self) -> np.ndarray:
        """Generates an n*m matrix of ranks one row at a time, where each
        position from M has the value -1"""

        replace_indices = sorted(zip(self.coo_M.row, self.coo_M.col))
        replace_indicies_index = 0

        for i, row in enumerate(matrix_mult(self.song_vecs, self.user_vecs)):
            # remove the listening counts from each row
            for j, val in enumerate(row):
                if (i, j) == replace_indices[replace_indicies_index]:
                    row[j] = -1
                    replace_indicies_index += 1

            yield convert_row_to_rank(row, i, self.coo_M.shape[0])

    @timeit(bold=True)
    def time_get_rank_matrix(self):
        print(f"{sum(1 for _ in lmf.get_rank_matrix())} rows")

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
        scale=10 ** scale, fp="triplets.csv"
    )

    lmf = LogisticMF(M, n_latent_factors=5, alpha=2,
                     l2_regularization=1, gamma=0.5, iterations=5)
    lmf.train(partition_size=(500, 500))
    print(lmf.log_likelihood())
    lmf.time_get_rank_matrix()
