import logging
import os
from savify import Savify
from savify.types import Type, Format, Quality
from savify.utils import PathHolder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Savify
s = Savify(
    api_credentials=(
        os.getenv('SPOTIFY_CLIENT_ID'),
        os.getenv('SPOTIFY_CLIENT_SECRET')
    ),
    quality=Quality.BEST,              # WORST, Q32K, Q96K, Q128K, Q192K, Q256K, Q320K, BEST
    download_format=Format.MP3,        # MP3, AAC, FLAC, M4A, OPUS, VORBIS, WAV
    path_holder=PathHolder(downloads_path='downloads'),
    group='%artist%/%album%',
    skip_cover_art=False
)

if __name__ == "__main__":
    playlist_url = "https://open.spotify.com/playlist/5qqOGm9AT1goFeiKcEF3NU"
    s.download(playlist_url, query_type=Type.PLAYLIST)