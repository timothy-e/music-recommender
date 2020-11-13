import pandas as pd

# remove duplicate columns

if __name__ == "__main__":
    
    path_to_track_data = "track_data.csv"
    
    track_data = pd.read_csv(path_to_track_data, encoding='latin-1')
    
    columns_to_drop = ['track_analysis_sample_rate', 'tatum_duration_variance', 'segment_duration_variance', 
                       'segment_loudness_max_variance', 'segment_loudness_start_variance', 'segment_loudness_end_variance',
                       'segment_pitches_variance', 'segment_timbre_variance', 'sections_duration_variance', 'sections_loudness_variance',
                       'sections_tempo_variance', 'sections_key_variance', 'title', 'artist']
    
    # Note: these columns seem to be sampled wrongly, will drop these columns for now
    columns_to_expand = ['segment_pitches_median', 'segment_pitches_stdev', 'segment_timbre_median', 'segment_timbre_stdev']

    columns_to_drop.extend(columns_to_expand)
    track_data.drop(columns = columns_to_drop)