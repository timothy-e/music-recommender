import json
import csv
from typing import Dict, Set, Tuple
import statistics
import pandas
from scipy.stats import chi2_contingency
import os
import spotipy
from pathlib import Path
from spotipy.oauth2 import SpotifyClientCredentials
from concurrent.futures import ThreadPoolExecutor
import concurrent
from multiprocessing.dummy import Pool

SongPair = Tuple[str, str]


def get_track_ids():
    df = pandas.read_csv('track_data.csv', encoding='ISO-8859-1')
    track_ids = df['song_id'].unique()
    return set(track_ids)


def filter_track_info(collab_track_ids: Set[str]):
    df = pandas.read_csv('track_data.csv', encoding='ISO-8859-1')
    filtered_df = df.loc[df.loc[:, "song_id"].isin(collab_track_ids)]

    print(len(df))
    print(len(filtered_df))
    
    with open('filtered_tracks.csv', 'w+') as f:
        filtered_df.to_csv(f, index=False)


def get_collab_ids():
    df = pandas.read_csv('filtered_triplets.txt')
    track_ids = df.iloc[:, 1].unique()
    return set(track_ids)


def filter_triplets(known_tracks: Set[str]):
    df = pandas.read_csv('train_triplets.txt', sep='\s+')
    filtered_df = df.loc[df.iloc[:, 1].isin(known_tracks)]
    filtered_df.columns = ['user_id', 'song_id', 'play_count']
    
    print(len(df))
    print(len(filtered_df))

    with open('filtered_triplets.txt', 'w+') as f:
        filtered_df.to_csv(f, index=False)

def get_collaborative_dataset(start=0, count=-1) -> Dict[str, SongPair]:
    song_info: Dict[str, SongPair] = dict()
    with open("echo_tracks.txt", "r", encoding="utf-8") as f:
        for i in range(start):
            next(f)
        for i, line in enumerate(f):
            if i == count:
                return song_info
            track_id, song_id, artist, title = line.strip("\n").split("<SEP>")
            song_info[song_id] = (title, artist)

    return song_info

    # song_ids: Set[str] = set()
    # with open("train_triplets.txt", "r") as f:
    #     for line in f:
    #         user_id, song_id, listen = line.split("\t")
    #         song_ids.add(song_id)

    # return dict((song_id, song_info[song_id]) for song_id in song_ids)


if __name__ == "__main__":
    # known_tracks = get_track_ids()
    # filter_triplets(known_tracks)

    known_tracks = get_collab_ids()
    filter_track_info(known_tracks)