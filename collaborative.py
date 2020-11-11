import pandas
from scipy.sparse import coo_matrix, lil_matrix
import sys
from typing import List, Tuple
from utils import timeit

ORIGINAL_DATASET_SIZE = 44120615


@timeit(bold=True)
def get_collab_matrix(scale=1, fp="triplets.csv") \
        -> Tuple[List[str], List[str], coo_matrix]:
    """Return `n` user_ids, `m` track_ids, and an `n x m` sparse matrix"""

    @timeit()
    def open_csv() -> pandas.DataFrame:
        nrows = int(ORIGINAL_DATASET_SIZE * scale)
        df = pandas.read_csv(fp, nrows=nrows)
        print(f"Working with {nrows} rows")
        return df

    df = open_csv()

    user_indices = {}
    track_indices = {}

    @timeit()
    def read_values() -> Tuple[List[str], List[str]]:
        """Return a list of user IDs and a list of track IDs, that will
        correspond to the rows/cols of the matrix"""
        user_ids = df.iloc[:, 0].unique()
        track_ids = df.iloc[:, 1].unique()

        for i, user_id in enumerate(user_ids):
            user_indices[user_id] = i
        for i, track_id in enumerate(track_ids):
            track_indices[track_id] = i
        return user_ids, track_ids

    user_ids, track_ids = read_values()

    @timeit()
    def create_matrix() -> coo_matrix:
        """Converts the list of triplets into a sparse matrix"""
        # construct as a list of list matrix because it's way easier
        M = lil_matrix((len(user_ids), len(track_ids)))

        for _, row in df.iterrows():
            user_ind = user_indices[row["user_id"]]
            track_ind = track_indices[row["song_id"]]

            M[user_ind, track_ind] = int(row["play_count"])
        return M.tocoo()  # save as coordinate matrix

    return user_ids, track_ids, create_matrix()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    scale = -arg.count("s")
    user_labels, track_labels, M = get_collab_matrix(scale=10 ** scale)
