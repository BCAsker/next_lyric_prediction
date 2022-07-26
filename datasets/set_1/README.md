# Information about dataset 1

## song_data.tsv
The 3 columns are as follows, with the first two being a multi-index:
* Song name
* Song artist
* Song ID (Ascending by track name)

## test.tsv and train.tsv

Test and train files are in .tsv format. Tabs separate columns, newlines separate rows.
Song 1 and song 2 can be any songs in the database, including the same song. These are the 7 columns: 
* Song ID 1
* Number of line in song 1
* Line from song 1
* Song ID 2
* Number of line in song 2
* Lyric 2
* Does lyric 1 follow lyric 2?

Rules:
* Each line number will appear precisely once in the database
* Line numbers start from 0 (for convenience)

### Notes
Some lines will be found many times in the database, possibly even across different songs. This is particularly 
prevalent with the blank line between verses. To account for this, we need to check the lyrics themselves when 
generating the x follows y values, not just the line number and song title