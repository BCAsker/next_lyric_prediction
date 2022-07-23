import config
import oauth_config
from generate_dataset_1 import main as generate_dataset

import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import argparse
import lyricsgenius
import socket


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

    return df.reset_index()


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
        return stripped_lyrics.replace("\n\n\n", "\n\n")
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


def main():
    args = parse_args()

    spotify = create_spotify_client(args.spotify_client_id, args.spotify_client_secret)
    genius = create_genius_client(args.genius_token)

    # Can be useful for debugging
    pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)

    df = generate_playlist_dataframe(spotify, args.playlist_id, genius)
    x = generate_dataset(df)
    print(x)


if __name__ == "__main__":
    main()
