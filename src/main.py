import re

import config
import oauth_config
import torch
from generate_dataset_1 import main as generate_dataset
from dataset_1 import Dataset1
from model_basic_bert import BasicBertPredictor

import os
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import argparse
import lyricsgenius
import socket
from torch.utils.data import DataLoader
from transformers import BertTokenizer

DATAFRAME_HEADERS = ["track_name", "artist_name", "lyrics"]


def generate_playlist_dataframe(spotify: spotipy.Spotify, playlist_id, genius: lyricsgenius.Genius):
    df = pd.DataFrame(columns=DATAFRAME_HEADERS)

    # Get the playlist tracks and artists
    playlist_items = spotify.playlist_items(playlist_id, limit=config.SPOTIPY_DEFAULT_LIMIT)
    while playlist_items["next"] is not None:
        track_names = [item["track"]["name"] for item in playlist_items["items"]]
        artists = [item["track"]["artists"][0]["name"] for item in playlist_items["items"]]
        lyrics = [None] * len(playlist_items["items"])

        new_rows = pd.DataFrame(
            {
                DATAFRAME_HEADERS[0]: track_names,
                DATAFRAME_HEADERS[1]: artists,
                DATAFRAME_HEADERS[2]: lyrics,
            }
        )
        df = pd.concat([df, new_rows])
        playlist_items = spotify.next(playlist_items)

    # Get the lyrics for the songs
    df["lyrics"] = df.apply(get_lyrics, axis=1, result_type="expand", genius=genius)
    df["stripped_lyrics"] = df["lyrics"].apply(tidy_genius_lyrics)

    return df.reset_index(drop=True)


def get_lyrics(df_row, genius):
    raw_data_path = get_raw_data_path(df_row["track_name"], df_row["artist_name"])
    if os.path.exists(raw_data_path):
        with open(raw_data_path, "rb") as f:
            return f.read().decode("utf8")
    else:
        lyrics = ""
        remaining_tries = 3
        while remaining_tries > 0:
            try:
                fetched = genius.search_song(df_row["track_name"], df_row["artist_name"], get_full_info=False)
                if fetched is not None:
                    lyrics = fetched.to_text()
                with open(raw_data_path, "wb") as f:
                    f.write(lyrics.encode("utf8"))
                break
            except socket.timeout:
                remaining_tries -= 1
                print(f"Timed out, attempting {remaining_tries} more time(s)")

        return lyrics


def get_raw_data_path(track_name, artist_name):
    # TODO: Improve replacements
    return os.path.join(os.pardir, "raw_data", f"{artist_name} {track_name}.txt").replace("/", "").replace("?", "")


def tidy_genius_lyrics(annotated_lyrics, remove_annotations=True):
    if annotated_lyrics is not None:
        # Drop the first line which is always just the title
        stripped_lyrics = "\n".join(annotated_lyrics.split("\n")[1:])

        if remove_annotations:
            # Annotations have the form "[Chorus]", "[Verse 1]", etc.
            while "[" in stripped_lyrics:
                start = stripped_lyrics.index("[")
                end = stripped_lyrics.index("]")
                stripped_lyrics = stripped_lyrics[:start] + stripped_lyrics[end + 1 :]

        # Originals can end with "Embed" for some reason so remove
        if stripped_lyrics.endswith("Embed"):
            stripped_lyrics = stripped_lyrics[:-5]

        # Make sure there aren't any double blank lines left by the above
        out = re.sub("\n\n+", "\n\n", stripped_lyrics)
        # Clear out any leading newlines
        while out.startswith("\n"):
            out = out[1:]
        # Clear out any trailing numbers (which genius seems to occasionaly give us for some reason)
        while out.endswith(tuple([str(i) for i in range(0, 10)])):
            out = out[:-1]
        return out
    else:
        return None


# TODO: Scrape some chords to go with our lyrics from ultimate guitar
def scrape_chords():
    return NotImplementedError()


def create_spotify_client(client_id, client_secret):
    spotify_auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=oauth_config.SPOTIFY_REDIRECT_URI,
        scope=oauth_config.SPOTIFY_SCOPES,
    )
    return spotipy.Spotify(auth_manager=spotify_auth)


def create_genius_client(token):
    return lyricsgenius.Genius(token)


def parse_args():
    parser = argparse.ArgumentParser(description="Get the lyrics for a spotify playlist")
    parser.add_argument("spotify_client_id", type=str, help="The client ID for the Spotify app")
    parser.add_argument("spotify_client_secret", type=str, help="The client secret for the Spotify app")
    parser.add_argument("genius_token", type=str, help="The token for the Genius app")
    parser.add_argument(
        "playlist_id",
        type=str,
        help="The id of the playlist to download the lyrics from",
    )

    return parser.parse_args()


def generate_dataloaders_from_dfs(train_df, test_df):
    batch_size = 4
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    tokenized_training_data = bert_tokenizer(
        list(train_df["line_prompt"]),
        list(train_df["line_query"]),
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=768,
        return_tensors="pt"
    )
    tokenized_test_data = bert_tokenizer(
        list(test_df["line_prompt"]),
        list(test_df["line_query"]),
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=768,
        return_tensors="pt",
    )

    train_dataset = Dataset1(tokenized_training_data, torch.LongTensor(list(train_df["follows"])))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Dataset1(tokenized_test_data, torch.LongTensor(list(test_df["follows"])))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def train(train_dataloader, model):
    print("Beginning to train")

    # TODO: Learn about / experiment with different loss functions and optimsers
    loss_funct = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.00001)

    # TODO: Make epochs const
    for epoch in range(1):
        epoch_loss = 0

        for batch_index, (inputs, labels) in enumerate(train_dataloader):
            optimiser.zero_grad()
            outputs = model(*inputs)
            loss = loss_funct(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()

            if batch_index % 25 == 0:
                print(f"Loss after batch {batch_index}/{len(train_dataloader)} in epoch {epoch} is {epoch_loss}")

    print("Training complete")


def test(test_dataloader, model):
    print("Beginning to test")

    pred_batches = []
    lab_batches = []

    for batch_index, (inputs, labels) in enumerate(test_dataloader):
        result = model(*inputs)
        preds = torch.argmax(result, dim=1)

        pred_batches.append(preds)
        lab_batches.append(labels)

    print("Testing complete")
    return torch.cat(pred_batches, dim=0), torch.cat(lab_batches, dim=0)


def print_statistics(predictions, labels):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0

    for pred, lab in zip(predictions, labels):
        if pred == 0:
            if lab == 0:
                true_neg += 1
            elif lab == 1:
                false_neg += 1
        elif pred == 1:
            if lab == 0:
                false_pos += 1
            elif lab == 1:
                true_pos += 1

    accuracy = (true_pos + true_neg) / len(predictions)
    precision = true_pos / (true_pos + true_neg) if true_pos + true_neg > 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_pos > 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0

    print(f"Total: {len(predictions)}, TP: {true_pos}, TN: {true_neg}, FP: {false_pos}, FN: {false_neg}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")


def main():
    args = parse_args()

    spotify = create_spotify_client(args.spotify_client_id, args.spotify_client_secret)
    genius = create_genius_client(args.genius_token)

    # Can be useful for debugging
    pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)

    df = generate_playlist_dataframe(spotify, args.playlist_id, genius)
    train_df, test_df = generate_dataset(df)
    train_dataloader, test_dataloader = generate_dataloaders_from_dfs(train_df, test_df)
    model = BasicBertPredictor()

    train(train_dataloader, model)
    preds, labs = test(test_dataloader, model)
    print_statistics(preds, labs)


if __name__ == "__main__":
    main()
