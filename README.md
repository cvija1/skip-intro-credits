# 🎬 Skip Intro and credits

Automatically detect TV show intros and end credits using audio fingerprinting.

This project provides a Python command line tool that identifies recurring intro and credits segments across video files. It enables features similar to **“Skip Intro”** and **“Skip Credits”** found in platforms like Plex or Jellyfin.

The system leverages:

- **Chromaprint (fpcalc)** for audio fingerprint generation
- **FFmpeg / ffprobe** for audio extraction and duration detection
- A **sliding window similarity algorithm** for robust matching

It is resilient to compression differences, small timing offsets, and minor audio quality variations.

---

## 🚀 Features

- 🔎 Detect shared intro sequences
- 🎞 Detect recurring end credits
- 📂 Compare one source file against multiple targets
- 📁 Automatically analyze all `.mp4` files in a folder
- ⚙ Fully customizable detection parameters
- 📦 Structured JSON output for easy integration

---

## 🛠 Requirements

### Python
- Python **3.10+**
- `numpy`

Install dependency:

```bash
pip install numpy
```

### FFmpeg

Required for audio extraction and duration detection.

Install:

**Ubuntu/Debian**
```bash
sudo apt install ffmpeg
```

**macOS**
```bash
brew install ffmpeg
```

**Windows**
Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH.

Verify installation:

```bash
ffmpeg -version
ffprobe -version
```

### Chromaprint (fpcalc)

Used for generating audio fingerprints.

Install:

**Ubuntu/Debian**
```bash
sudo apt install libchromaprint-tools
```

**macOS**
```bash
brew install chromaprint
```

**Windows**
Download from https://acoustid.org/chromaprint and add `fpcalc` to PATH.

Verify:

```bash
fpcalc -version
```

---

## 📦 Installation

```bash
git clone https://github.com/cvija1/skip-intro-credits.git
cd skip-intro-credits
pip install numpy
```

Ensure `ffmpeg`, `ffprobe`, and `fpcalc` are accessible from your terminal.

---

## 🧪 Usage

Basic command:

```bash
python main.py source_video.mp4 target1.mp4 target2.mp4
```

### Arguments

| Argument | Description |
|----------|------------|
| `source` | Reference video file |
| `targets` | One or more target files to compare |
| `--folder PATH` | Analyze all `.mp4` files in folder |
| `--begin` | Analyze only intros |
| `--end` | Analyze only credits |

---

## 💡 Examples

### Compare intros and credits across specific episodes

```bash
python main.py thrones-01.mp4 thrones-02.mp4 thrones-03.mp4
```

### Analyze only intros from a folder

```bash
python main.py thrones-01.mp4 --folder "/path/to/episodes" --begin
```

### Detect only credits

```bash
python main.py thrones-01.mp4 thrones-02.mp4 --end
```

---

## 📤 Example Output

The script prints detailed match information during execution and outputs a final JSON summary:

```json
[
  {
    "source_file": "thrones-01.mp4",
    "target_file": "thrones-02.mp4",
    "intro_start": null,
    "intro_end": 32,
    "credits_start": 1282,
    "credits_end": 1299
  },
  {
    "source_file": "thrones-01.mp4",
    "target_file": "thrones-03.mp4",
    "intro_start": null,
    "intro_end": 33,
    "credits_start": 1285,
    "credits_end": 1300
  }
]
```

### Notes

- All intervals are expressed in **seconds**
- `null` indicates:
  - Intro starting at 0
  - Credits ending too close to total duration
  - If all fields are null then there is no matching

---

## 🧠 How It Works

### 1️⃣ Audio Extraction

- Intro detection fingerprints the beginning directly
- Credits detection extracts the final N seconds using FFmpeg

### 2️⃣ Fingerprint Generation

Chromaprint generates a numeric signature representing the audio.

### 3️⃣ Sliding Window Matching

- Source fingerprint is divided into windows (e.g. 6 seconds)
- Each window slides across the target fingerprint
- Best alignment above similarity threshold is recorded

### 4️⃣ JSON Aggregation

Results are structured per target for integration into:

- Media servers
- Databases
- Automation pipelines

---

## ⚙ Configuration

Edit these parameters at the top of the script:

```python
SAMPLE_LENGTH_SEC = 300
WINDOW_SEC = 6
STEP_SEC = 1
MIN_SIMILARITY = 0.82
MIN_OVERLAP_POINTS = 8
MAX_WORKERS = 4
```

To adjust merging tolerance, modify the merge offset inside `merge_overlapping_intervals`.

---

## ⚠ Limitations

- Command-line only (no GUI)
- Performance scales with:
  - Sample length
  - Number of target files
- Optimized for `.mp4` (extend logic for other formats if needed)

---

## 🤝 Contributing

Pull requests are welcome.

If you have feature ideas, improvements, or bug reports, open an issue.

---

## 📜 License

MIT License.

---

## 🎧 Built With

- Chromaprint
- FFmpeg

Inspired by open-source