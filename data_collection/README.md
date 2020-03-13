## Data Collection Module

The data collection module is a script intended to pull transcript data from every video in a YouTube Playlist.
Example usage is below:

```
collect_transcripts.py collect_transcripts.py -p PLAYLIST_ID -o OUTPUT_FILE.txt -t YOUTUBE_API_TOKEN
```

### Required Arguments

* `-p / --playlist`
    * The YouTube playlist ID of the list containing the videos you wish to process.
* `-o / --output`
    * The path to the output file you wish to write.  If the file already exists, data will be appended.
* `-t / --token`
    * Your YouTube API token
    
### Potential Improvements

* Allowing for token storage in environmental variables or elsewhere
* Output alternatives