"""
Script that obtains transcript data from all of the videos in a YouTube playlist

Example:
    collect_transcripts.py -p PLAYLIST_ID -o OUTPUT_FILE.txt -t YOUTUBE_API_TOKEN

Todo:
    * Currently outputting (potentially very large) text files.  Shall we do something else?
"""


import dateutil.parser
import requests
from argparse import ArgumentParser
from html import unescape
from typing import Dict
from youtube_transcript_api import YouTubeTranscriptApi


def get_video_info(playlist: str, token: str) -> Dict[str, Dict[str, str]]:
    req_url = "https://www.googleapis.com/youtube/v3/playlistItems"
    videos = {}
    cont = True
    next_page = None
    count = 0

    print("Gathering...")

    # Results from this API endpoint are paginated.  So, here we go.
    while cont:
        if next_page:
            params = {
                "part": "snippet",
                "key": token,
                "playlistId": playlist,
                "maxResults": 50,
                "pageToken": next_page,
            }
        else:
            params = {
                "part": "snippet",
                "key": token,
                "playlistId": playlist,
                "maxResults": 50,
            }

        results = requests.get(req_url, params=params)

        if results.json().get("items"):
            items = results.json().get("items")

            for item in items:
                if not videos.get(item["snippet"]["resourceId"]["videoId"]):
                    # We don't really need this info currently, but it's there and maybe it will come
                    # in handy for something down the line
                    videos[item["snippet"]["resourceId"]["videoId"]] = {
                        "date": dateutil.parser.parse(item["snippet"]["publishedAt"]),
                        "title": unescape(item["snippet"]["title"])
                    }

                    count += 1

        next_page = results.json().get("nextPageToken")
        print(f"Collected {count} videos")

        if not next_page:
            print("Finished collection")
            cont = False

    return videos


def write_output(videos: Dict[str, Dict[str, str]], output_file: str) -> None:
    with open(output_file, "a", encoding="utf-8") as file:
        count = 0
        write_text = ""

        print("Writing file.  This may take a while...")

        for video in list(videos.keys()):
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video)
                full_text = "\n".join([entry["text"] for entry in transcript])
                write_text += f"{full_text}\n"
                count += 1
            except Exception:
                # The transcript API throws many exceptions, so ultimately this just
                # ignores all the videos with problems, no transcripts, etc.
                pass

            if count % 10 == 0:
                print(f"{count} transcripts written")
                file.write(f"{write_text}\n")
                write_text = ""


# Main execution starts here
if __name__ == "__main__":
    # Set up the arg parser and grab the arguments
    parser = ArgumentParser(description="Collect transcripts from the videos in a YouTube playlist.")
    req_parser = parser.add_argument_group("Required Arguments")
    req_parser.add_argument("-p", "--playlist", help="Desired playlist ID.", required=True)
    req_parser.add_argument("-o", "--output", help="File (.txt) to which transcripts are output.", required=True)
    req_parser.add_argument("-t", "--token", help="YouTube API Token", required=True)
    args = parser.parse_args()

    # With that down, we can grab the info of the videos in the playlist:
    vid_info = get_video_info(args.playlist, args.token)

    # Then we can grab the transcripts and write the file
    write_output(vid_info, args.output)

    # We are done
    print("Operation complete.  Enjoy!")
