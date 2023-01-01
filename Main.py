from Recommender import Recommender


def get_diff_from_int(diff):
    if diff == 0:
        return 'Easy'
    elif diff == 1:
        return 'Normal'
    elif diff == 2:
        return 'Hard'
    elif diff == 3:
        return 'Expert'
    elif diff == 4:
        return 'Expert+'


if __name__ == '__main__':
    dataset_path = 'datasets/beatsaversongs.csv'  # Scraped from api.beatsaver.com on 27/12/2022

    # Recommender parameters.
    n_recommendations = 20
    n_best_tags = 3

    # Dancy song:
    song_id = '1af89'
    difficulty = 3
    characteristic = 0
    # Speed song:
    # song_id = '12b62'
    # difficulty = 4
    # characteristic = 0
    # Tech song:
    # song_id = '2b0c9'
    # difficulty = 4
    # characteristic = 0

    print(
        f'{n_recommendations} recommendations for https://beatsaver.com/maps/{song_id} {get_diff_from_int(difficulty)}')

    r = Recommender(dataset_path=dataset_path)
    recommendations = r.recommend(song_id=song_id, difficulty=difficulty, characteristic=0,
                                  n_recommendations=n_recommendations, n_best_tags=n_best_tags)
    for r in recommendations:
        print(
            f"\tMap link: https://beatsaver.com/maps/{r.song_id} (Char: {r.characteristic} - "
            f"{get_diff_from_int(r.difficulty)}) "
            f"(m{round(r.meta_sim * 100)}%/t{round(r.tag_sim * 100)}%/{round(r.total_sim * 100)}%)")
