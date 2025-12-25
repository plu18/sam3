import os
import subprocess

# Keywords to search for
# Updated for Traffic and Pedestrians
KEYWORDS = [
    "pedestrians crossing street real time",
    "busy crosswalk shibuya 4k",
    "city street walking view",
    "traffic intersection pedestrians",
]

# Output directory
OUTPUT_DIR = "my_experiments/data/sports_videos"

# Number of videos to download per keyword
NUM_VIDEOS_PER_KEYWORD = 2


def download_videos():
    """
    Downloads videos using yt-dlp based on keywords.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print(f"Starting download of sports videos to {OUTPUT_DIR}...")
    print(f"Keywords: {KEYWORDS}")

    for keyword in KEYWORDS:
        print(f"\n--- Processing keyword: '{keyword}' ---")

        # yt-dlp command construction
        # ytsearchN: searches for N videos
        search_query = f"ytsearch{NUM_VIDEOS_PER_KEYWORD}:{keyword}"

        cmd = [
            "yt-dlp",
            search_query,
            "-o",
            f"{OUTPUT_DIR}/%(title)s.%(ext)s",
            # Prefer mp4 for compatibility
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            # Limit file size to avoid filling up disk
            "--max-filesize",
            "50M",
            # Limit duration to keep videos short (e.g., < 1 min)
            "--match-filter",
            "duration < 60",
            "--no-playlist",
            # Do not overwrite existing files
            "--no-overwrites",
            # Ignore errors and continue
            "--ignore-errors",
        ]

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading for keyword '{keyword}': {e}")
        except FileNotFoundError:
            print(
                "Error: yt-dlp not found. Please install it using 'pip install yt-dlp' or 'apt install yt-dlp'."
            )
            return

    print(f"\nDownload process completed. Check {OUTPUT_DIR} for videos.")


if __name__ == "__main__":
    download_videos()
