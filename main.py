import subprocess
import numpy as np
import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor


SAMPLE_LENGTH_SEC   = 300
WINDOW_SEC          = 6
STEP_SEC            = 1
MIN_SIMILARITY      = 0.80
MIN_OVERLAP_POINTS  = 8
MAX_WORKERS = 4


def merge_overlapping_intervals(matches):
    if not matches:
        return None, None

    intervals = sorted(matches, key=lambda x: x['tgt_start'])

    merged = []
    curr_start = intervals[0]['tgt_start']
    curr_end   = intervals[0]['tgt_end']

    for interv in intervals[1:]:
        if interv['tgt_start'] <= curr_end + 3: 
            curr_end = max(curr_end, interv['tgt_end'])
        else:
            merged.append((curr_start, curr_end))
            curr_start = interv['tgt_start']
            curr_end   = interv['tgt_end']

    merged.append((curr_start, curr_end))

    if not merged:
        return None, None

    best_start, best_end = max(merged, key=lambda x: x[1] - x[0])
    start = int(best_start)               
    end   = int(best_end)  

    if end - start < 5:
        return None, None

    return start, end


def get_video_duration(filename):
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filename
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        return float(out)
    except Exception as e:
        print(f"Warning: could not get length of file {filename} → {e}")
        return None

def get_fingerprint_beginning(filename, length_sec):
    cmd = ['fpcalc', '-raw', '-length', str(length_sec), filename]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        idx = out.find('FINGERPRINT=') + 12
        if idx < 12:
            raise ValueError(f"Could not parse fingerprint for {filename}")
        fp_str = out[idx:].strip()
        if not fp_str:
            raise ValueError("Empty fingerprint")
        return list(map(int, fp_str.split(',')))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"fpcalc error {filename}:\n{e.output}")


def get_fingerprint_end(filename, length_sec):
    ffmpeg_cmd = [
        'ffmpeg',
        '-sseof', f'-{length_sec}',
        '-i', filename,
        '-vn',
        '-ac', '1',
        '-ar', '22050',
        '-f', 'wav',
        'pipe:1'
    ]

    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    fpcalc = subprocess.Popen(
        ['fpcalc', '-raw', '-length', str(length_sec), '-'],
        stdin=ffmpeg.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    ffmpeg.stdout.close()
    out, _ = fpcalc.communicate()

    idx = out.find('FINGERPRINT=') + 12
    if idx < 12:
        raise RuntimeError("Could not parse fingerprint")

    fp_str = out[idx:].strip()
    return list(map(int, fp_str.split(',')))



def correlation(fp1, fp2):
    if not fp1 or not fp2:
        return 0.0
    min_len = min(len(fp1), len(fp2))
    if min_len < MIN_OVERLAP_POINTS:
        return 0.0
    cov = sum(32 - (a ^ b).bit_count() for a, b in zip(fp1, fp2))
    return cov / (min_len * 32.0)


def find_best_offset(fp_window, fp_target):
    best_sim = 0.0
    best_offset = 0
    max_offset = len(fp_target) - len(fp_window) + 1
    search_step = max(1, len(fp_window) // 8)

    for offset in range(0, max_offset, search_step):
        segment = fp_target[offset:offset + len(fp_window)]
        sim = correlation(fp_window, segment)
        if sim > best_sim:
            best_sim = sim
            best_offset = offset
    return best_sim, best_offset


def compute_fp_for_segment(args):
    filename, from_end = args
    fp_func = get_fingerprint_end if from_end else get_fingerprint_beginning
    return (filename, from_end), fp_func(filename, SAMPLE_LENGTH_SEC)


def prepare_all_fingerprints(source, targets, do_begin=True, do_end=True):
    segments = []
    if do_begin:
        segments.append(False)  
    if do_end:
        segments.append(True)   

    all_files = [source] + targets
    all_tasks = [(f, seg) for f in all_files for seg in segments]

    fp_cache = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for key, fp in executor.map(compute_fp_for_segment, all_tasks):
            fp_cache[key] = fp

    return fp_cache


def compare_target_with_source(source_file, target_file, fp_cache, durations, from_end):
    fp_source = fp_cache[(source_file, from_end)]
    fp_target = fp_cache[(target_file, from_end)]

    source_duration = durations[source_file]
    target_duration = durations[target_file]

    window_points = int(WINDOW_SEC * (len(fp_source) / SAMPLE_LENGTH_SEC))
    step_points   = int(STEP_SEC   * (len(fp_source) / SAMPLE_LENGTH_SEC))

    matches = []
    pos = 0
    while pos + window_points <= len(fp_source):
        window = fp_source[pos:pos + window_points]
        sim, offset = find_best_offset(window, fp_target)

        rel_start_src = pos * SAMPLE_LENGTH_SEC / len(fp_source)
        rel_end_src   = (pos + window_points) * SAMPLE_LENGTH_SEC / len(fp_source)
        rel_start_tgt = offset * SAMPLE_LENGTH_SEC / len(fp_target)
        rel_end_tgt   = (offset + window_points) * SAMPLE_LENGTH_SEC / len(fp_target)

        if from_end:
            s_src = (source_duration + (-SAMPLE_LENGTH_SEC + rel_start_src)) if source_duration else (-SAMPLE_LENGTH_SEC + rel_start_src)
            e_src = (source_duration + (-SAMPLE_LENGTH_SEC + rel_end_src))   if source_duration else (-SAMPLE_LENGTH_SEC + rel_end_src)
            s_tgt = (target_duration + (-SAMPLE_LENGTH_SEC + rel_start_tgt)) if target_duration else (-SAMPLE_LENGTH_SEC + rel_start_tgt)
            e_tgt = (target_duration + (-SAMPLE_LENGTH_SEC + rel_end_tgt))   if target_duration else (-SAMPLE_LENGTH_SEC + rel_end_tgt)
        else:
            s_src = rel_start_src
            e_src = rel_end_src
            s_tgt = rel_start_tgt
            e_tgt = rel_end_tgt

        if sim >= MIN_SIMILARITY:
            matches.append({
                'sim': round(sim, 4),
                'tgt_start': round(s_tgt, 1),
                'tgt_end':   round(e_tgt, 1),
            })

        pos += step_points
    return os.path.basename(target_file), matches


def analyze_file_against_targets(source_file, target_files, fp_cache, durations, from_end=False):
    section_name = "Credits" if from_end else "Intro"
    print(f"{section_name} analysing..")
    fp_source = fp_cache[(source_file, from_end)]
    window_points = int(WINDOW_SEC * (len(fp_source) / SAMPLE_LENGTH_SEC))
    step_points   = int(STEP_SEC   * (len(fp_source) / SAMPLE_LENGTH_SEC))
    source_duration = durations[source_file]
    all_matches = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(compare_target_with_source, source_file, target_file, fp_cache, durations, from_end)
            for target_file in target_files
        ]

        for future in futures:
            target_name, matches = future.result()
            all_matches[target_name] = matches

            if not matches:
                print(f"\n→ {target_name}: No segments ≥ {MIN_SIMILARITY}")
    return all_matches

def fetch_duration(filename):
    return filename, get_video_duration(filename)

def fetch_duration_parallel(source,targets):
    all_files = [source] + targets
    durations = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_duration, f) for f in all_files]
        for future in futures:
            filename, dur = future.result()
            durations[filename] = dur
    return durations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare intro and credits segments between video files using audio fingerprinting"
    )
    parser.add_argument(
        "source",
        help="Source (reference) video file, e.g. thrones-01.mp4"
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="One or more target video files to compare against the source (optional if --folder is used)"
    )
    parser.add_argument(
        "--folder", "--dir",
        help="Folder from which ALL .mp4 files will be taken as targets (source file is excluded if present)"
    )
    parser.add_argument(
        "--end",
        action="store_true",
        help="Analyze only the end of the videos (credits)"
    )
    parser.add_argument(
        "--begin",
        action="store_true",
        help="Analyze only the beginning of the videos (intro)"
    )
    args = parser.parse_args()

    source = os.path.abspath(args.source) 

    if args.folder:
        folder_path = os.path.abspath(args.folder)
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not valid dir!")
            exit(1)

        all_mp4 = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith('.mp4') and os.path.isfile(os.path.join(folder_path, f))]

        targets = [f for f in all_mp4 if f != source] 

        if not targets:
            print(f"There is no .mp4 files in dir {folder_path} (except maybe source)")
            exit(1)

        print(f"Founded {len(targets)} .mp4 files in dir {folder_path}")
    else:
        targets = [os.path.abspath(t) for t in args.targets]
        if not targets:
            print("Error: You must specify at least one target file or use the --folder option.")
            exit(1)

    do_begin = args.begin or not args.end
    do_end   = args.end or not args.begin
    fp_cache = prepare_all_fingerprints(source, targets, do_begin, do_end)
    durations = fetch_duration_parallel(source, targets)
    final_results = []

    intro_all_matches = None
    credits_all_matches = None

    if do_begin:
        intro_all_matches = analyze_file_against_targets(source, targets, fp_cache, durations, from_end=False)

    if do_end:
        credits_all_matches = analyze_file_against_targets(source, targets, fp_cache, durations, from_end=True)

    for target in targets:
        target_name = os.path.basename(target)

        res = {
            "source_file": os.path.basename(source),
            "target_file": target_name,
            "intro_start": None,
            "intro_end":   None,
            "credits_start": None,
            "credits_end":   None
        }

        if intro_all_matches and target_name in intro_all_matches:
            matches = intro_all_matches[target_name]
            if matches:
                intervals = [(m['tgt_start'], m['tgt_end']) for m in matches if m['tgt_start'] is not None]
                if intervals:
                    start, end = merge_overlapping_intervals([{'tgt_start': s, 'tgt_end': e} for s, e in intervals])
                    res["intro_start"] = start
                    res["intro_end"]   = end

        if credits_all_matches and target_name in credits_all_matches:
            matches = credits_all_matches[target_name]
            if matches:
                intervals = [(m['tgt_start'], m['tgt_end']) for m in matches if m['tgt_start'] is not None]
                if intervals:
                    start, end = merge_overlapping_intervals([{'tgt_start': s, 'tgt_end': e} for s, e in intervals])
                    res["credits_start"] = start
                    res["credits_end"]   = end

        final_results.append(res)

    for res in final_results:
        target_file = [t for t in targets if os.path.basename(t) == res["target_file"]][0]
        target_duration = durations[target_file]

        if target_duration is not None:
            if res["intro_start"] is not None and res["intro_start"] <= 0.5:
                res["intro_start"] = None

            if (res["credits_end"] is not None and 
                target_duration - res["credits_end"] <= 3):
                res["credits_end"]   = None
        else:
            print(f"Warning: Cannot get duration of {res['target_file']} – skipping credits null check")
    print("FINAL RESULT")
    print(json.dumps(final_results, indent=2, ensure_ascii=False))