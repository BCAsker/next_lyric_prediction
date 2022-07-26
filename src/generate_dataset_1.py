import os
import random
import pandas as pd

TRAIN_PATH = os.path.join(os.pardir, "datasets", "set_1", "train.tsv")
TEST_PATH = os.path.join(os.pardir, "datasets", "set_1", "test.tsv")
SONG_DATA_PATH = os.path.join(os.pardir, "datasets", "set_1", "song_ids.tsv")


def get_lines_df(df: pd.DataFrame):
    lines = df["stripped_lyrics"].apply(str.split, sep="\n").apply(enumerate).apply(list).explode()
    lines = pd.DataFrame(lines.tolist(), columns=["line_num", "line"], index=lines.index)
    names = df.drop(["lyrics", "stripped_lyrics"], axis=1)

    return pd.merge(names, lines, how="outer", left_index=True, right_index=True).reset_index(drop=True)


def get_song_id_df(df: pd.DataFrame):
    if os.path.exists(SONG_DATA_PATH) and os.stat(SONG_DATA_PATH).st_size > 0:
        song_id_df = pd.read_csv(SONG_DATA_PATH, sep="\t", index_col=["track_name", "artist_name"])
    else:
        song_id_df = df[["track_name", "artist_name"]]
        song_id_df = song_id_df.set_index(["track_name", "artist_name"])
        song_id_df.sort_index(inplace=True)
        song_id_df["song_id"] = range(len(df))
        song_id_df.to_csv(SONG_DATA_PATH, sep="\t")

    lyric_df = df.merge(song_id_df, left_on=["track_name", "artist_name"], right_index=True).drop(
        ["track_name", "artist_name"], axis=1
    )
    return lyric_df, song_id_df


def get_paired_line_df(lines_df: pd.DataFrame):
    order = list(lines_df.index)

    # Drop an element if there are an odd number of lines
    if len(order) % 2 == 1:
        order = order[1:]

    random.shuffle(order)

    first = order[: len(order) // 2]
    second = order[len(order) // 2 :]

    prompt_df = lines_df[["song_id", "line_num", "line"]].iloc[first].reset_index(drop=True)
    query_df = lines_df[["song_id", "line_num", "line"]].iloc[second].reset_index(drop=True)

    return prompt_df.merge(query_df, left_index=True, right_index=True, suffixes=["_prompt", "_query"])


def does_line_2_follow_line_1(row, lines_df: pd.DataFrame):
    # If row is the last line in the song, return false straight away because nothing can follow it
    if (
        row["line_num_prompt"]
        == lines_df[["song_id", "line_num"]].groupby("song_id").max().loc[row["song_id_prompt"]].iloc[0]
    ):
        return False
    else:
        line_following_prompt = (
            lines_df["line"]
            .loc[(lines_df["song_id"] == row["song_id_prompt"]) & (lines_df["line_num"] == row["line_num_prompt"]) + 1]
            .iloc[0]
        )
        return line_following_prompt == row["line_query"]


def save_test_train_split(paired_df: pd.DataFrame, lines_df: pd.DataFrame):
    paired_df["follows"] = paired_df.apply(does_line_2_follow_line_1, axis=1, lines_df=lines_df)
    split_index = (len(paired_df) * 4) // 5
    train_df = paired_df.iloc[:split_index]
    test_df = paired_df.iloc[split_index:]

    train_df.to_csv(TRAIN_PATH, sep="\t", index_label="ID")
    test_df.to_csv(TEST_PATH, sep="\t", index_label="ID")

    return train_df, test_df


def load_test_train_split():
    train_df = pd.read_csv(TRAIN_PATH, sep="\t", index_col="ID")
    test_df = pd.read_csv(TEST_PATH, sep="\t", index_col="ID")

    return train_df, test_df


def get_train_test_dfs(paired_df, lines_df):
    if (
        os.path.exists(TRAIN_PATH)
        and os.path.exists(TEST_PATH)
        and os.stat(TRAIN_PATH).st_size > 0
        and os.stat(TEST_PATH).st_size > 0
    ):
        return load_test_train_split()
    else:
        return save_test_train_split(paired_df, lines_df)


def main(df):
    lyric_df, song_id_df = get_song_id_df(df)
    lines_df = get_lines_df(lyric_df)
    paired_df = get_paired_line_df(lines_df)
    return get_train_test_dfs(paired_df, lines_df)
