# Information about dataset 1

## song_data.tsv
The 4 columns are as follows:
* Song ID
* Song name
* Song artist
* List of unique lyric IDs (see below) describing the song

## test.tsv and train.tsv

Test and train files are in .tsv format. Tabs separate columns, newlines separate rows.
Song 1 and song 2 can be any songs in the database, including the same song. These are the 7 columns: 
* Song ID 1
* Unique Line number 1 (see below)
* Line from song 1
* Song ID 2
* Unique Line number 2 (see below)
* Lyric 2
* Does lyric 1 follow lyric 2?

Rules:
* Each line number will appear precisely once in the database
* Test and train datasets will be stratified
* Line numbers start from 0 (for convenience)

## Unique lyric IDs
Songs often have the same line (including the empty line, splitting verse/choruses) repeated multiple times. Obviously,
we only really care if the lyrics are the same, not if the line came from the first chorus or the second. Thus, we
represent songs as follows:

* Give each unique line a unique number. If a line is repeated, we give it the same ID as its first occurrence.
* Represent songs as a list of unique lines.

For example:
1 This is the first line
2 This is the second line
3 This is another line
3 This is another line
4 
5 That was an empty line
4 
5 That was an empty line
3 This is another line

So the song is [1, 2, 3, 3, 4, 5, 4, 5, 3], where each index is a line number