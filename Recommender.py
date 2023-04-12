import numpy as np
from numpy.linalg import norm
import pandas as pd

from models.Recommendation import Recommendation


class Recommender:
    """
        This recommender system does only look at the metadata of the songs,
        not the actually mapping (No pattern recognition).

        The metadata also contains tags,
        using jaccard similarity the songs that don't have (enough) similar tags get eliminated.
        Cosine similarity is used to further optimize the recommendations on its metadata.

        Maps without tags are the least accurate because they purly rely on metadata similarity which isn't accurate.
        Tag prediction or adding tags is not in this Recommender. (but could be added in the dataset manually)
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.songs = self.get_songs(self.dataset_path)

    def recommend(self, song_id, difficulty, characteristic=0, n_recommendations=20, n_best_tags=3):
        # Get the latest songs.
        # Might be a good idea to get the latest songs from an in-memory database
        # that polls the newest songs from the beatsaver api.
        songs = self.songs

        # Columns that are not used for metadata recommendation.
        non_meta_columns = ['BeatMapId', 'UploaderName', 'MapName', 'CoverUrl', 'Tags', 'TagSim']

        # Get tags and add the uploader as tag of the given song by song_id and difficulty.
        compare_tags = songs.loc[songs.BeatMapId == song_id].loc[songs.Difficulty == difficulty].loc[
            songs.Characteristic == characteristic].Tags.unique()[0].split(';')

        # Calculate the similarity of the tags of all songs.
        songs['TagSim'] = [self.jaccard_similarity(compare_tags, tags.split(';')) for tags in songs.Tags]

        # Get the best scoring tags to use for song elimination.
        vals = list(songs.TagSim.value_counts().index)
        vals.sort(reverse=True)

        # Get the metadata of the given song.
        compare_meta = songs.loc[songs.BeatMapId == song_id].loc[songs.Difficulty == difficulty].loc[
            songs.Characteristic == characteristic].drop(
            columns=non_meta_columns)

        # Eliminate songs that have a tag similarity lower than best n scoring tags.
        songs_with_tags = songs.loc[songs.TagSim.isin(vals[:n_best_tags])].drop(columns=non_meta_columns)

        # Normalize the metadata of the songs because the distance of notes (big number) would create a higher bias.
        songs_with_tags = (songs_with_tags - songs_with_tags.min()) / (songs_with_tags.max() - songs_with_tags.min())
        songs_with_tags = songs_with_tags.fillna(0)

        # Calculate the cosine similarity of the metadata of all songs.
        songs_with_tags['MetaSim'] = [self.cosine_sim(compare_meta.values[0], row) for row in songs_with_tags.values]

        # Restore the original data.
        songs_with_tags['BeatMapId'] = songs.loc[songs.TagSim.isin(vals[:n_best_tags])].BeatMapId
        songs_with_tags['MapName'] = songs.loc[songs.TagSim.isin(vals[:n_best_tags])].MapName
        songs_with_tags['CoverUrl'] = songs.loc[songs.TagSim.isin(vals[:n_best_tags])].CoverUrl
        songs_with_tags['UploaderName'] = songs.loc[songs.TagSim.isin(vals[:n_best_tags])].UploaderName
        songs_with_tags['Difficulty'] = songs.loc[songs.TagSim.isin(vals[:n_best_tags])].Difficulty
        songs_with_tags['Characteristic'] = songs.loc[songs.TagSim.isin(vals[:n_best_tags])].Characteristic
        songs_with_tags['TagSim'] = songs.loc[songs.TagSim.isin(vals[:n_best_tags])].TagSim
        songs_with_tags['TotalSimilarity'] = (songs_with_tags.MetaSim + songs_with_tags.TagSim) / 2

        # Sort by total similarity and get n best recommendations. Cast results into tuple.
        recs = [Recommendation(song_id=r.BeatMapId, difficulty=r.Difficulty, characteristic=r.Characteristic,
                               meta_sim=r.MetaSim, tag_sim=r.TagSim, total_sim=r.TotalSimilarity)
                for r in songs_with_tags.sort_values(by='TotalSimilarity').itertuples() if
                r.BeatMapId != song_id]

        # Return the recommendations. Reverse because the best recommendations are at the end.
        return list(reversed(recs))[:n_recommendations]

    def get_songs(self, dataset_path):
        songs = pd.read_csv(dataset_path, delimiter=',')
        songs.BeatMapId = songs.BeatMapId.astype('string')
        songs.IsRanked = songs.IsRanked.astype(int)
        songs.IsCurated = songs.IsCurated.astype(int)
        songs.HasChroma = songs.HasChroma.astype(int)
        songs.HasCinema = songs.HasCinema.astype(int)
        songs.HasMappingExtensions = songs.HasMappingExtensions.astype(int)
        songs.HasNoodleExtensions = songs.HasNoodleExtensions.astype(int)
        songs.Tags = songs.apply(lambda r: self.add_uploader_to_tags(str(r.Tags).split(';'), r.UploaderName), axis=1)
        songs.Tags = songs.Tags.apply(lambda r: ';'.join(r))
        songs.Tags = songs.Tags.astype('string')

        # Dealing with outliers. Not deleting but replacing them with an “A lot” equivalent
        # to keep the items relevant as someone might be interested in these outliers.
        songs.loc[songs.Njs > 40, "Njs"] = 40
        songs.loc[songs.Njs < -10, "Njs"] = -10
        songs.loc[songs.Offset > 20, "Offset"] = 20
        songs.loc[songs.Offset < -20, "Offset"] = -20
        songs.loc[songs.Notes > 10000, 'Notes'] = 10000
        songs.loc[songs.Events > 20000, 'Events'] = 20000
        songs.loc[songs.Bombs > 5000, 'Bombs'] = 5000
        songs.loc[songs.Obstacles > 10000, 'Obstacles'] = 10000
        songs.loc[songs.Errors > 20, 'Errors'] = 20
        songs.loc[songs.Resets > 20, 'Resets'] = 20
        songs.loc[songs.Warns > 20, 'Warns'] = 20
        songs.loc[songs.Bpm > 1000, 'Bpm'] = 1000
        songs.loc[songs.Duration > 1800, 'Duration'] = 1800
        return songs

    @staticmethod
    def add_uploader_to_tags(tags, uploader):
        if len(tags) > 0 and tags[0] != 'nan':
            tags.append(uploader)
        return tags

    # Manually implementing similarity functions to gain performance.
    @staticmethod
    def jaccard_similarity(a, b) -> float:
        a = set(a)
        b = set(b)
        j = float(len(a.intersection(b))) / len(a.union(b))
        return j

    @staticmethod
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
