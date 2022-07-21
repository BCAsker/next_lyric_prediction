import os
import random
import pandas as pd

TRAIN_PATH = os.path.join(os.curdir, "datasets", "set_1", "train.tsv")
TEST_PATH = os.path.join(os.curdir, "datasets", "set_1", "test.tsv")


def enumerate_and_get_unique_line_id(split_song):
    unique_lines = {}
    next_id = 0
    ids = []

    for line in split_song:
        if line not in unique_lines.keys():
            unique_lines[line] = next_id
            next_id += 1
        ids.append(unique_lines[line])

    return zip(range(len(split_song)), ids, split_song)


def get_lines_df(df: pd.DataFrame):
    lines = (
        df["stripped_lyrics"].apply(str.split, sep="\n").apply(enumerate_and_get_unique_line_id).apply(list).explode()
    )
    lines = pd.DataFrame(lines.tolist(), columns=["line_num", "unique_line_num", "line"], index=lines.index)
    names = df.drop(["lyrics", "stripped_lyrics"], axis=1)

    return (
        pd.merge(names, lines, how="outer", left_index=True, right_index=True)
        .drop("index", axis=1)
        .reset_index(drop=True)
    )


def create_paired_line_df(lines_df: pd.DataFrame):
    order = list(lines_df.index)

    # Drop an element if there are an odd number of lines
    if len(order) % 2 == 1:
        order = order[1:]

    random.shuffle(order)

    first = order[: len(order) // 2]
    second = order[len(order) // 2 :]

    prompt_df = lines_df[["track_name", "artist_name", "unique_line_num", "line"]].iloc[first].reset_index(drop=True)
    query_df = lines_df[["track_name", "artist_name", "unique_line_num", "line"]].iloc[second].reset_index(drop=True)

    return prompt_df.merge(query_df, left_index=True, right_index=True, suffixes=["_prompt", "_query"])


def save_test_train_split(paired_df: pd.DataFrame):
    split_index = (len(paired_df) * 4) // 5
    train_df = paired_df.iloc[:split_index]
    test_df = paired_df.iloc[split_index:]

    train_df.to_csv(TRAIN_PATH, sep="\t")
    test_df.to_csv(TEST_PATH, sep="\t")

    return train_df, test_df


def load_test_train_split():
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    test_df = pd.read_csv(TEST_PATH, sep="\t")

    return train_df, test_df


def get_train_test_dfs(paired_df):
    with open(TRAIN_PATH) as f_train:
        with open(TEST_PATH) as f_test:
            files_populated = len(f_train.read()) > 0 and len(f_test.read()) > 0

    if files_populated:
        return load_test_train_split()
    else:
        return save_test_train_split(paired_df)


def main(df):
    lines_df = get_lines_df(df)
    paired_df = create_paired_line_df(lines_df)
    return get_train_test_dfs(paired_df)
