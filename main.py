import subprocess
import numpy as np
import os
import json
import argparse


SAMPLE_LENGTH_SEC   = 600
WINDOW_SEC          = 6
STEP_SEC            = 1
MIN_SIMILARITY      = 0.80
MIN_OVERLAP_POINTS  = 8


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
        print(f"Upozorenje: ne mogu dobiti dužinu fajla {filename} → {e}")
        return None

def get_fingerprint_beginning(filename, length_sec):
    cmd = ['fpcalc', '-raw', '-length', str(length_sec), filename]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        idx = out.find('FINGERPRINT=') + 12
        if idx < 12:
            raise ValueError(f"Ne može parsirati fingerprint za {filename}")
        fp_str = out[idx:].strip()
        if not fp_str:
            raise ValueError("Prazan fingerprint")
        return list(map(int, fp_str.split(',')))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"fpcalc greška za početak {filename}:\n{e.output}")


def get_fingerprint_end(filename, length_sec):
    import tempfile

    temp_dir = tempfile.gettempdir()
    temp_wav = os.path.join(temp_dir, f"last_{length_sec}s_{os.path.basename(filename)}.wav")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-err_detect', 'ignore_err',
        '-sseof', f'-{length_sec}',
        '-i', filename,
        '-vn', '-ac', '1', '-ar', '22050',
        '-t', str(length_sec),
        temp_wav
    ]

    try:
        subprocess.check_output(ffmpeg_cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg error {filename}:\n{e.output}")

    if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 10000:
        raise RuntimeError(f"WAV is not created or is small len: {temp_wav}")

    try:
        cmd = ['fpcalc', '-raw', '-length', str(length_sec), temp_wav]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        idx = out.find('FINGERPRINT=') + 12
        if idx < 12:
            raise ValueError("could not parse fingerprint")
        fp_str = out[idx:].strip()
        if not fp_str:
            raise ValueError("Empty fingerprint")
        return list(map(int, fp_str.split(',')))
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass


def correlation(fp1, fp2):
    if not fp1 or not fp2:
        return 0.0
    min_len = min(len(fp1), len(fp2))
    if min_len < MIN_OVERLAP_POINTS:
        return 0.0
    cov = sum(32 - bin(a ^ b).count('1') for a, b in zip(fp1, fp2))
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


def analyze_file_against_targets(source_file, target_files, from_end=False):
    section_name = "credits" if from_end else "intro"
    print(f"\n{'='*60}")
    print(f" {section_name.upper()} ANALYSIS  – source: {os.path.basename(source_file)}")
    print(f"{'='*60}")

    fp_source_func = get_fingerprint_end if from_end else get_fingerprint_beginning
    fp_source = fp_source_func(source_file, SAMPLE_LENGTH_SEC)

    window_points = int(WINDOW_SEC * (len(fp_source) / SAMPLE_LENGTH_SEC))
    step_points   = int(STEP_SEC   * (len(fp_source) / SAMPLE_LENGTH_SEC))


    source_duration = get_video_duration(source_file)
    all_matches = {}
    for target_file in target_files:
        print(f"\n  → Compare with: {os.path.basename(target_file)}")

        fp_target = fp_source_func(target_file, SAMPLE_LENGTH_SEC)

        target_duration = get_video_duration(target_file)

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
                # Apsolutno vreme od početka
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
                    'tgt_start': round(s_src, 1),
                    'tgt_end':   round(e_src, 1),
                    'tgt_start': round(s_tgt, 1),
                    'tgt_end':   round(e_tgt, 1),
                })

                src_str = f"{s_src:7.1f}s" if not from_end else f"{s_src:7.1f}s"
                tgt_str = f"{s_tgt:7.1f}s" if not from_end else f"{s_tgt:7.1f}s"
                print(f"    Match {sim:.4f} | "
                      f"{os.path.basename(source_file)}: {src_str} – {e_src:7.1f}s  →  "
                      f"{os.path.basename(target_file)}: {tgt_str} – {e_tgt:7.1f}s")

            pos += step_points

        all_matches[os.path.basename(target_file)] = matches
        if matches:
            print(f"    → Found {len(matches)} segment ≥ {MIN_SIMILARITY}")
        else:
            print(f"    → No segmenats ≥ {MIN_SIMILARITY}")
    return all_matches

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

    final_results = []

    intro_all_matches = None
    credits_all_matches = None

    if do_begin:
        intro_all_matches = analyze_file_against_targets(source, targets, from_end=False)

    if do_end:
        credits_all_matches = analyze_file_against_targets(source, targets, from_end=True)

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
        target_duration = get_video_duration(target_file)

        if target_duration is not None:
            if res["intro_start"] is not None and res["intro_start"] <= 0.5:
                res["intro_start"] = None

            if (res["credits_end"] is not None and 
                target_duration - res["credits_end"] <= 3):
                res["credits_end"]   = None
        else:
            print(f"Warning: Cannot get duration of {res['target_file']} – skipping credits null check")

    print("\n" + "="*80)
    print(" FINAL RESULT (array of JSON objects) ".center(80, "="))
    print("="*80)
    print(json.dumps(final_results, indent=2, ensure_ascii=False))