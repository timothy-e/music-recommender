# Music Recommender

We want to find out how to introduce users to new songs that they enjoy listening to, and evaluate if it is more successful to suggest songs based on what other users with similar tastes listen to (collaborative filtering strategy), or suggest songs that have similar qualities to songs that the user likes (content-based filtering strategy). Both approaches try to answer the question of "closeness," but the data that is used differs. Collaborative filtering works on a large sample of (user, song) pairs, and content-based filtering requires an in-depth analysis of each song's acoustic features and metadata.

For our collaborative filtering approach, we implemented a Logistic Matrix Factorization probabilistic model using implicit user feedback.

For our content-based approach, we use three different metrics to find similarity between the user profile vector and song vectors. We will experiment with Euclidean distance, Cosine distance, and Pearson Correlation.


## Setting up the Environment

In the directory, to set up the virtual environment:

```
python3 -m venv .venv
```

Then to use the environment, run

```
source .venv/bin/activate
```

or

```
.\.venv\bin\activate.bat
```

to enter the virtual environment.

Use

```
pip install -r requirements.txt
```

to get the requirements and

```
deactivate
```

to exit the virtual environment.

## Performance

Run
```
python performance.py [-s]
```

The number of `s`s in the flag indicate how small of a dataset we want. No `-s` is the whole dataset, `-s` is one tenth, `-ss` is one hundredth, ...

This will initilize, train, and print out the MPR score of the two recommender schemes (see below).

Open the file to change tuning parameters or update `k` in `k`-cross-fold validation.

## Collaborative Filtering

Run

```
python collaborative.py [-s]
```

Running this file will read in the collaborative matrix and store it as a sparse matrix.

Running

```
python logistic_mf.py [-s]
```

will print the log likelihood of the trained matrix, and also create a rank matrix.

## Content Filtering

Run

```
python content_rec.py

```

to try out the content recommender. It uses both the `triplets` and `track_data` datasets. Running this file doesn't print anything, but it's an easy way to verify that the code runs.