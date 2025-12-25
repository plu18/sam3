import os

VIDEO_DIR = "my_experiments/data/sports_videos"

# Mapping from keywords in filename to category
CATEGORY_KEYWORDS = {
    "football": ["football", "soccer"],
    "golf": ["golf", "swing", "els", "fleetwood", "driver", "iron", "putt", "tracer"],
    "gymnastics": ["gymnastics", "floor", "ohashi", "biles"],
    "badminton": ["badminton"],
    "tennis": ["tennis", "nadal", "djokovic", "open", "rally"],
    "skateboarding": ["skateboarding", "board"],
    "basketball": ["basketball", "nba", "lakers", "celtics"],
    "parkour": ["parkour"],
    "pedestrian": [
        "pedestrian",
        "crosswalk",
        "walking",
        "street",
        "intersection",
        "shibuya",
        "crossing",
        "road rules",
    ],
}


def get_category(filename):
    filename_lower = filename.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return category
    return "other"


def rename_videos():
    if not os.path.exists(VIDEO_DIR):
        print(f"Directory not found: {VIDEO_DIR}")
        return

    files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    # Sort to ensure deterministic order
    files.sort()

    counters = {}

    # First pass: find max index for existing renamed files
    for filename in files:
        for cat in list(CATEGORY_KEYWORDS.keys()) + ["other"]:
            if filename.startswith(cat + "_") and filename[len(cat) + 1 : -4].isdigit():
                index = int(filename[len(cat) + 1 : -4])
                if cat not in counters:
                    counters[cat] = index
                else:
                    counters[cat] = max(counters[cat], index)

    print(f"Renaming videos in {VIDEO_DIR}...")
    print(f"Current counters: {counters}")

    for filename in files:
        # Skip files that already match the pattern category_number.mp4
        # But allow re-processing of 'other' category in case we improved detection
        is_already_renamed = False
        for cat in list(CATEGORY_KEYWORDS.keys()):  # Exclude "other" from this check
            if filename.startswith(cat + "_") and filename[len(cat) + 1 : -4].isdigit():
                is_already_renamed = True
                break
        if is_already_renamed:
            # print(f"Skipping '{filename}' (already renamed)")
            continue

        category = get_category(filename)

        if category not in counters:
            counters[category] = 0

        counters[category] += 1

        index = counters[category]
        new_name = f"{category}_{index}.mp4"

        old_path = os.path.join(VIDEO_DIR, filename)
        new_path = os.path.join(VIDEO_DIR, new_name)

        # Handle collision if new_path already exists (e.g. from a previous run)
        if os.path.exists(new_path):
            print(
                f"Warning: Target file '{new_name}' already exists. Skipping '{filename}'."
            )
            continue

        print(f"Renaming '{filename}' -> '{new_name}'")
        os.rename(old_path, new_path)


if __name__ == "__main__":
    rename_videos()
