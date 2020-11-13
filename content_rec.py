import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

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
                            export_as_csv = True,
                            export_path = "user_profile_mat.csv") :
    
    user_median_play_count = (triplets_data.groupby(by=["user_id"])[["play_count"]]
                                           .median())
    user_profile_mat = []
    user_profile_mat_columns = (track_data.select_dtypes(include = ['number'])
                                          .columns)
    
    for tuple in user_median_play_count.itertuples():
        user_id = tuple.Index
        median_threshold = tuple.play_count
        user_triplets = triplets_data.loc[user_id]
        filtered_songids = user_triplets.loc[(user_triplets.play_count >= median_threshold), ['song_id']]
        filtered_songs = pd.merge(track_data, filtered_songids, how='inner', on="song_id")
        user_profile = filtered_songs.mean(axis=0)
        user_profile_mat.append(user_profile)
    
    user_profile_mat_df = pd.DataFrame(user_profile_mat, 
                                       columns=user_profile_mat_columns,
                                       index=user_median_play_count.index)
    
    if (export_as_csv):
        user_profile_mat_df.to_csv(export_path, index=True)
    
    return user_profile_mat_df
    


    """[Compute similarity matrix for each user song pair given user profile matrix and track_data]
    function: 
        1. train_model: 
            Return: 
                S, a similarity matrix of size [n_users, n_songs] such that S_{ij} is the similarity measures between
                user_i and song_j
        2. get_rank_matrix
    """

class contentbasedRec:
    
    def __init__(self, 
                 user_profile_df : pd.core.frame.DataFrame, 
                 track_df: pd.core.frame.DataFrame,
                 user_song_count_mat,
                 similarity_measures = ["cosine", "euclidean", "pearson"]):
        
        if (not user_profile_df.columns.equals(track_df.columns)):
            raise Exception("contentbasedRec: user_profile_df's columns does not match track_dfs")
        
        self.user_profile_df = user_profile_df
        self.track_df = track_df
        self.user_song_count_mat = user_song_count_mat
        self.similarity_measures = similarity_measures
        
    self.cosine_similarity_mat = None
    self.euclid_similarity_mat = None
    self.pearson_similarity_mat = None
    
    self.cosine_rank_mat = None
    self.euclid_rank_mat = None
    self.pearson_rank_mat = None
    
    def train_model(self):
        user_profile_mat = self.user_profile_df.to_numpy()
        track_mat = self.track_df.to_numpy()
        
        if "cosine" in self.similarity_measures:
            self.cosine_similarity_mat = pairwise_distances(X = user_profile_mat, 
                                                            Y = track_mat,
                                                            metric = "cosine")
            
        
        if "euclidean" in self.similarity_measures:
            self.euclid_similarity_mat = pairwise_distances(X = user_profile_mat, 
                                                            Y = track_mat,
                                                            metric = "euclidean")
            
        if "pearson" in self.similarity_measures:
            self.euclid_similarity_mat = pairwise_distances(X = user_profile_mat, 
                                                            Y = track_mat,
                                                            metric = "correlation")
        
    def get_rank_matrix(similarity_measure):
        if similarity_measure == "cosine":
            return self.cosine_rank_mat
        elif similarity_measure == "euclidean":
            return self.euclid_rank_mat
        elif similarity_measure == "pearson":
            return self.pearson_rank_mat
        else :
            raise Exception("contentbasedRec: unsupported similarity measure:", similarity_measure)

        """[Get User Song Count Matrix with entries = 1 if user-song pair has listening count > 0]
        NOTE: The order of rows follows the order of user_id in user_profile_mat_df and order of columns follows the order of song_ids in track_data
        """
        
def get_user_song_count_mat(user_profile_mat_df, 
                            track_data,
                            triplets_data):
    
    user_ids = user_profile_mat_df.index
    user_ids_dict = {user_id: index[0] for index, user_id in np.ndenumerate(user_ids)}
    song_ids = track_data.index
    song_ids_dict = {song_id: index[0] for index, song_id in np.ndenumerate(song_ids)}
    user_song_count_mat = np.zeros(shape=(len(user_ids),len(song_ids)))
    
    for triplet in triplets_data.itertuples():
        user_id = triplet.user_id
        song_id = triplet.song_id
        user_id_idx = user_ids_dict[user_id]
        song_id_idx = song_ids_dict[song_id]
        user_song_count_mat[user_id_idx, song_id_idx] = 1
    
    return user_song_count_mat
    

    """[Run content based recommender]
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
                                                      export_as_csv = True,
                                                      export_path = "user_profile_mat.csv")
        
        