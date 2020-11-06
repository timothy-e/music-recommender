from typing import Dict, Tuple
import statistics
import pandas
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from multiprocessing.dummy import Pool

SongPair = Tuple[str, str]


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


def search_for_song(spotify, title, artist):
    retry = 3
    while True:
        try:

            results = spotify.search(q=f'{title} {artist}', type='track')
            if results['tracks']['items']:
                return results['tracks']['items'][0]['id']
        except Exception:
            if retry == 0:
                return None
            retry -= 1
            print(f"Search failed on {title} {artist}")
        if "(" in title and ")" in title and "[" in title and "]" in title:
            start_round = title.rindex('(')
            end_round = title.rindex(')')
            start_square = title.rindex('[')
            end_square = title.rindex(']')
            if start_round < start_square:
                if end_square <= start_square:
                    return None
                title = title[:start_square] + title[end_square + 1:]
            else:
                if end_round <= start_round:
                    return None
                title = title[:start_round] + title[end_round + 1:]
        elif "(" in title and ")" in title:
            start = title.rindex('(')
            end = title.rindex(')')
            if end <= start:
                return None
            title = title[:start] + title[end + 1:]
        elif "[" in title and "]" in title:
            start = title.rindex('[')
            end = title.rindex(']')
            if end <= start:
                return None
            title = title[:start] + title[end + 1:]
        else:
            return None


def get_analysis(spotify, spid):
    analysis = spotify.audio_analysis(spid)
    if not analysis:
        return
    del analysis["meta"]
    del analysis["track"]["codestring"]
    del analysis["track"]["code_version"]
    del analysis["track"]["echoprintstring"]
    del analysis["track"]["echoprint_version"]
    del analysis["track"]["synchstring"]
    del analysis["track"]["synch_version"]
    del analysis["track"]["rhythmstring"]
    del analysis["track"]["rhythm_version"]
    del analysis["track"]["sample_md5"]
    del analysis["track"]["num_samples"]
    del analysis["track"]["duration"]
    del analysis["track"]["offset_seconds"]
    del analysis["track"]["window_seconds"]
    del analysis["track"]["analysis_channels"]
    del analysis["track"]["end_of_fade_in"]
    del analysis["track"]["start_of_fade_out"]
    del analysis["bars"]
    del analysis["beats"]

    track = analysis.pop("track")
    for k, v in track.items():
        analysis[f"track_{k}"] = v
    del track

    tatums = analysis.pop("tatums")
    analysis["tatum_duration_median"] = statistics.median(
        [t["duration"] for t in tatums])
    analysis["tatum_duration_stdev"] = statistics.stdev(
        [t["duration"] for t in tatums])
    analysis["tatum_duration_variance"] = statistics.variance(
        [t["duration"] for t in tatums])
    del tatums

    segments = analysis.pop("segments")
    for var in ["duration", "loudness_max", "loudness_start", "loudness_end"]:
        values = [s[var] for s in segments]
        analysis[f"segment_{var}_median"] = statistics.median(values)
        analysis[f"segment_{var}_stdev"] = statistics.stdev(values)
        analysis[f"segment_{var}_variance"] = statistics.variance(values)
    pitches = list(zip(*[s["pitches"] for s in segments]))
    analysis["segment_pitches_median"] = [
        statistics.median(pitch_class) for pitch_class in pitches]
    analysis["segment_pitches_stdev"] = [
        statistics.stdev(pitch_class) for pitch_class in pitches]
    analysis["segment_pitches_variance"] = [
        statistics.variance(pitch_class) for pitch_class in pitches]
    del pitches
    timbres = list(zip(*[s["timbre"] for s in segments]))
    analysis["segment_timbre_median"] = [statistics.median(t) for t in timbres]
    analysis["segment_timbre_stdev"] = [statistics.stdev(t) for t in timbres]
    analysis["segment_timbre_variance"] = [
        statistics.variance(t) for t in timbres]
    del timbres
    del segments

    sections = analysis.pop("sections")
    for var in ["duration", "loudness", "tempo", "key"]:
        values = [s[var] for s in sections]
        analysis[f"sections_{var}_median"] = statistics.median(values)
        analysis[f"sections_{var}_stdev"] = statistics.stdev(values)
        analysis[f"sections_{var}_variance"] = statistics.variance(values)
    analysis["section_mode_mean"] = statistics.mean(
        [s["mode"] for s in sections])
    del sections

    return analysis


def get_info(quintuple):
    sp, sid, title, artist, spid = quintuple
    try:
        return {
            "song_id": sid,
            "title": title,
            "artist": artist,
            **get_analysis(sp, spid)
        }
    except Exception:
        print(f"FAILED ON {title} by {artist} ({spid})")


def get_metadata(triple):
    song_id, title, artist = triple
    result = search_for_song(sp, title, artist)
    if not result:
        return None
    return (song_id, title, artist, result)


if __name__ == "__main__":
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    pool = Pool(50)

    for j in range(916550, 1000000, 50):
        collab_songs: Dict[str, SongPair] = get_collaborative_dataset(j, 50)
        songs_with_spid: Dict[str, Tuple[str, str, str]] = dict()

        song_triples = [(song_id, title, artist)
                        for song_id, (title, artist) in collab_songs.items()]
        song_quadruples = list(
            filter(None, pool.map(get_metadata, song_triples)))

        for sid, title, artist, spid in song_quadruples:
            songs_with_spid[sid] = (title, artist, spid)

        print(
            f"On round {j}: found {len(song_quadruples)}/{len(collab_songs)}.")

        del collab_songs

        song_quintuples = [(sp, sid, *rest) 
                           for sid, rest in songs_with_spid.items()]
        all_info = list(filter(None, pool.map(get_info, song_quintuples)))

        df = pandas.DataFrame(all_info)

        with open("track_data.csv", "a+") as f:
            df.to_csv(f, encoding='utf-16')
