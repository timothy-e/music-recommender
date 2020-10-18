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

SongPair = Tuple[str, str]


def get_collaborative_dataset() -> Dict[str, SongPair]:
    song_info: Dict[str, SongPair] = dict()
    with open("echo_tracks.txt", "r", encoding="utf-8") as f:
        for line in f:
            track_id, song_id, artist, title = line.strip("\n").split("<SEP>")
            song_info[song_id] = (title, artist)

    song_ids: Set[str] = set()
    with open("train_triplets.txt", "r") as f:
        for line in f:
            song_ids.add(song_id)

    return dict((song_id, song_info[song_id]) for song_id in song_ids)


def search_for_song(spotify, title, artist):
    while True:
        results = sp.search(q=f'{title} {artist}', type='track')
        if results['tracks']['items']:
            return results['tracks']['items'][0]['id']

        if "(" in title and ")" in title and "[" in title and "]" in title:
            start_round = title.rindex('(')
            end_round = title.rindex(')')
            start_square = title.rindex('[')
            end_square = title.rindex(']')
            if start_round < start_square:
                title = title[:start_square] + title[end_square + 1:]
            else:
                title = title[:start_round] + title[end_round + 1:]
        elif "(" in title and ")" in title:
            start = title.rindex('(')
            end = title.rindex(')')
            title = title[:start] + title[end + 1:]
        elif "[" in title and "]" in title:
            start = title.rindex('[')
            end = title.rindex(']')
            title = title[:start] + title[end + 1:]
        else:
            return None


if __name__ == "__main__":
    collab_songs: Dict[str, SongPair] = get_collaborative_dataset()

    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    songs_with_spid: Dict[str, Tuple[str, str, str]] = dict()

    found = 0
    not_found = set()
    for song_id, (title, artist) in collab_songs.items():
        result = search_for_song(sp, title, artist)
        if result:
            songs_with_spid[song_id] = (title, artist, result)
            found += 1
        else:
            not_found.add((title, artist))

    print(f"found {found}/{len(collab_songs)}. Examples of failures: ")
    for i, (song, artist) in enumerate(not_found):
        if i > 20:
            break
        print(song, artist)

    del collab_songs
    del not_found

    final = []
    songs = list((sid, title, artist, spid)
                 for sid, (title, artist, spid) in songs_with_spid.items())
    for i in range(0, len(songs), 50):
        # 50-item batches to save request time
        spids = [song[3] for song in songs[i:i+50]]
        features = sp.audio_features(spids)
        final.extend([
            {
                "song_id": sid,
                "title": title,
                "artist": artist,
                "features": json.load(features)
            }
            for (sid, title, artist, spid), feature
            in zip(songs[i:i+50], features)
        ])

    with open("track_data.csv", "w+") as f:
        json.dump(final, f)
