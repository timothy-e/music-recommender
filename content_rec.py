import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from collaborative import get_collab_matrix
from utils import convert_row_to_rank

"""[Create user profile matrix given triplets and track_data]
    NOTE: all columns in track_data must be of dtypes numerical (int or float)
    Argument:
        1. track_data_path:    file path to track_data
        2. triplets_data_path: file path to triplets_data
        3. export_as_csv:      export output as csv when set to TRUE
        4. export_path:        file path for exporting results to
    Return:
        n * f Pandas Dataframe,
        where n is the number of users and f is the number of song audio features
    """


def get_user_profile_matrix(track_data,
                            triplets_data,
                            export_as_csv=True,
                            export_path="user_profile_mat.csv"):

    user_median_play_count = (triplets_data.groupby(by=["user_id"])[["play_count"]]
                                           .median())
    user_profile_mat = []
    # remove titles, artist columns as they are non-numerical
    track_data.drop(columns=[
        'title', 'artist',
        'segment_pitches_median', 'segment_pitches_stdev',
        'segment_pitches_variance', 'segment_timbre_median',
        'segment_timbre_stdev', 'segment_timbre_variance'], inplace=True)
    user_profile_mat_columns = track_data.columns

    for tuple in user_median_play_count.itertuples():
        user_id = tuple.Index
        median_threshold = tuple.play_count
        user_triplets = triplets_data.loc[user_id]
        filtered_songids = user_triplets.loc[(
            user_triplets.play_count >= median_threshold)].song_id
        filtered_songs = track_data[track_data.index.isin(filtered_songids)]
        user_profile = filtered_songs.mean(
            axis=0, numeric_only=True, skipna=True)
        user_profile_mat.append(user_profile)

    user_profile_mat_df = pd.DataFrame(user_profile_mat,
                                       columns=user_profile_mat_columns,
                                       index=user_median_play_count.index)

    if (export_as_csv):
        user_profile_mat_df.to_csv(export_path, index=True)

    return user_profile_mat_df

    """[Compute similarity and rank matrices for each user song pair given user profile matrix and track_data]
    class function:
        1. train_model
        2. get_rank_matrix
        3. get_similarity_matrix
    """


class contentbasedRec:

    def __init__(self,
                 user_profile_df: pd.core.frame.DataFrame,
                 track_df: pd.core.frame.DataFrame,
                 user_song_count_idxmat,
                 similarity_measures=["cosine", "euclidean", "pearson"]):

        self.user_profile_df = user_profile_df
        self.track_df = track_df
        self.user_song_count_idxmat = user_song_count_idxmat
        self.similarity_measures = similarity_measures

        self.cosine_similarity_mat = None
        self.euclid_similarity_mat = None
        self.pearson_similarity_mat = None

        self.cosine_rank_mat = None
        self.euclid_rank_mat = None
        self.pearson_rank_mat = None

    def get_rank_matrix(self, metric):
        user_profile_mat = self.user_profile_df.to_numpy()
        track_mat = self.track_df.to_numpy()

        replace_indices = sorted(
            zip(self.user_song_count_idxmat.row, self.user_song_count_idxmat.col))
        replace_indicies_index = 0

        # TODO: `pairwise_distances` prob might have to be turned into a
        # generator but run it and see
        for i, row in enumerate(pairwise_distances(
            X=user_profile_mat,
            Y=track_mat,
            metric="correlation" if metric == "pearson" else metric
        )):
            # set observations to -1
            for j, val in enumerate(row):
                if (i, j) == replace_indices[replace_indicies_index]:
                    row[j] = -1
                    replace_indicies_index += 1

            yield convert_row_to_rank(row, i, user_profile_mat.shape[0])

    def get_similarity_matrix(self, similarity_measure):
        if similarity_measure == "cosine":
            return self.cosine_similarity_mat
        elif similarity_measure == "euclidean":
            return self.euclid_similarity_mat
        elif similarity_measure == "pearson":
            return self.pearson_similarity_mat
        else:
            raise Exception(
                "contentbasedRec: unsupported similarity measure:", similarity_measure)

    """[Get User Song Count Matrix with entries = True if user-song pair has listening count > 0, False otherwise]
    NOTE: The order of rows follows the order of user_id in user_profile_mat_df and order of columns follows the order of song_ids in track_data
          Used to turn all non-zero entries in the ranked matrices into -1
    """


def get_user_song_count_idxmat(user_profile_mat_df,
                               track_data,
                               triplets_data):

    user_ids = user_profile_mat_df.index
    user_ids_dict = {user_id: index[0]
                     for index, user_id in np.ndenumerate(user_ids)}
    song_ids = track_data.index
    song_ids_dict = {song_id: index[0]
                     for index, song_id in np.ndenumerate(song_ids)}
    user_song_count_idxmat = np.full((len(user_ids), len(song_ids)), False)

    for triplet in triplets_data.itertuples():
        user_id = triplet.Index
        song_id = triplet.song_id
        user_id_idx = user_ids_dict[user_id]
        song_id_idx = song_ids_dict[song_id]
        user_song_count_idxmat[user_id_idx, song_id_idx] = True

    return user_song_count_idxmat

    """[Run content based recommender]
    NOTE: Clean track_data before calling this function
    """


if __name__ == "__main__":

    """
    1. Transform songs and users into vectors of the same subspace
    2. Compute similarity matrix with one of the three similarity measures
    3. To recomnmend song, rank each song by their similarity score and take the top k songs
    """
    track_data_path = "mini_track_data.csv"
    triplets_data_path = "mini_triplets.csv"

    triplets_data = (pd.read_csv(triplets_data_path)
                       .set_index(["user_id"]))
    track_data = (pd.read_csv(track_data_path)
                    .set_index(["song_id"]))

    user_profile_mat_df = get_user_profile_matrix(track_data,
                                                  triplets_data,
                                                  export_as_csv=True,
                                                  export_path="user_profile_mat.csv")

    _, _, user_song_count_idxmat = get_collab_matrix(fp="mini_triplets.csv")
    print(user_song_count_idxmat)

    contentBasedModel = contentbasedRec(user_profile_mat_df,
                                        track_data,
                                        user_song_count_idxmat,
                                        similarity_measures=["cosine", "euclidean", "pearson"])

    contentBasedModel.train_model()
