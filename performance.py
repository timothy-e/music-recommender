import numpy as np
import sys
from scipy.sparse import coo_matrix, find
from logistic_mf import LogisticMF
from collaborative import get_collab_matrix


def cross_validation(M, k):
    # randomly divide nonzero entries in M into k sets
    # nonzero entries represented by 3 arrays
    (i, j, v) = find(M)  # i - user, j - song, v - listening count
    idx = np.arange(len(i))  # nonzero entries indices corresponding to i,j,v
    np.random.shuffle(idx)
    test_sets = np.array_split(idx, k)  # split entries into k subarrays
    avg_mpr = 0.0
    yikes = 0  # counts # of times we read in -1 rank ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    aiya = 0

    for test_set_i in test_sets:
        print(len(test_set_i))

        # convert test set to dict of user: (dict of song: listen_count)
        test_set = {}
        for entry in test_set_i:
            user = i[entry]
            song = j[entry]
            listen_count = v[entry]

            if user not in test_set:
                test_set[user] = {}
            test_set[user][song] = listen_count

        # create training set
        training_M = M.copy()
        for entry in test_set_i:
            # check we are setting correct values to 0
            if training_M[i[entry], j[entry]] != v[entry]:
                aiya += 1

            training_M[i[entry], j[entry]] = 0
        training_M = training_M.tocoo()

        # call logistic_mf w/ training_M
        lmf = LogisticMF(training_M, n_latent_factors=5, alpha=2,
                         l2_regularization=1, gamma=0.5, iterations=5)
        lmf.train(partition_size=(500, 500))

        # calculate MPR
        listen_sum = 0.0
        rank_sum = 0.0
        user = 0
        for user_song_ranks in lmf.get_rank_matrix():
            if user not in test_set:
                user += 1
                continue

            for song in test_set[user].keys():
                rank = user_song_ranks[song]
                if rank == -1:
                    yikes += 1  # ~~~~~~~~~~~~~~~~~~~~~~~~~

                listen_sum += test_set[user][song]
                rank_sum += test_set[user][song] * rank

            user += 1

        # fix MPR below
        print("yiiiikes x" + str(yikes))  # remove ~~~~~~~~~~~~~~~~~~~~~~~~~
        print("aiyaaa " + str(aiya))
        MPR = rank_sum / listen_sum
        print(rank_sum)
        print(listen_sum)
        print(MPR)
        avg_mpr += MPR
        break  # only testing one time rn

    return avg_mpr / k


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    scale = -arg.count("s")
    user_labels, track_labels, M = get_collab_matrix(
        scale=10 ** scale, fp="mid_triplets.csv"
    )
    print(cross_validation(M.tocsr(), 10))
