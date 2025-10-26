import os
import re
import csv
import json
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Set


def sanitize_name(name: str) -> str:
    s = name.strip().replace(' ', '_')
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def load_label_mappings(label_csv: str) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, str]]:
    mid_to_name: Dict[str, str] = {}
    name_to_idx: Dict[str, int] = {}
    name_to_mid: Dict[str, str] = {}
    with open(label_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            idx_str, mid, name = row[:3]
            name = name.strip().strip('"')
            try:
                idx = int(idx_str)
            except Exception:
                continue
            mid_to_name[mid] = name
            name_to_idx[name] = idx
            name_to_mid[name] = mid
    return mid_to_name, name_to_idx, name_to_mid


def parse_keep_list(keep_indices: Optional[str], keep_names: Optional[List[str]], label_csv: str) -> Tuple[List[str], List[str]]:
    mid_to_name, name_to_idx, name_to_mid = load_label_mappings(label_csv)
    keep_names_list: List[str] = []
    keep_mids_list: List[str] = []
    if keep_indices:
        idxs = [int(s) for s in keep_indices.split(',') if s.strip()]
        for name, idx in name_to_idx.items():
            if idx in idxs:
                keep_names_list.append(name)
                keep_mids_list.append(name_to_mid[name])
    elif keep_names:
        # keep_names is now a list; also support a single JSON-array string
        names: List[str] = []
        if isinstance(keep_names, (list, tuple)):
            for it in keep_names:
                if isinstance(it, str) and it.strip():
                    s = it.strip()
                    if s.startswith('[') and s.endswith(']'):
                        try:
                            arr = json.loads(s)
                            for nm in arr:
                                if isinstance(nm, str) and nm.strip():
                                    names.append(nm.strip())
                            continue
                        except Exception:
                            pass
                    names.append(s)
        elif isinstance(keep_names, str) and keep_names.strip():
            s = keep_names.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    arr = json.loads(s)
                    for nm in arr:
                        if isinstance(nm, str) and nm.strip():
                            names.append(nm.strip())
                except Exception:
                    pass
            else:
                names.append(s)
        for nm in names:
            mid = name_to_mid.get(nm)
            if mid:
                keep_names_list.append(nm)
                keep_mids_list.append(mid)
    else:
        # default: all classes
        for nm, mid in name_to_mid.items():
            keep_names_list.append(nm)
            keep_mids_list.append(mid)
    if not keep_mids_list:
        raise SystemExit('No classes selected for download')
    return keep_names_list, keep_mids_list


def read_segments(segments_csv: str) -> List[Tuple[str, float, float, List[str]]]:
    rows: List[Tuple[str, float, float, List[str]]] = []
    with open(segments_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#') or row[0] == 'YTID':
                continue
            try:
                ytid = row[0].strip()
                start = float(row[1])
                end = float(row[2])
                labels_str = row[3].strip().strip('"')
                mids = [s.strip() for s in labels_str.split(',') if s.strip()]
            except Exception:
                continue
            rows.append((ytid, start, end, mids))
    return rows


def ensure_deps():
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        raise SystemExit('Please install yt-dlp: pip install yt-dlp')
    from shutil import which
    if which('ffmpeg') is None:
        raise SystemExit('ffmpeg not found in PATH. Install ffmpeg and ensure it is available.')


def download_video_audio(ytid: str, tmp_dir: str, *, ua: Optional[str] = None, cookiefile: Optional[str] = None) -> Optional[str]:
    """Download best audio for a YouTube video using yt-dlp with 403 workarounds.
    Tries default client, then Android/IOS/TV clients to mitigate 403 Forbidden.
    Return path to downloaded file or None on failure.
    """
    import yt_dlp
    from yt_dlp.utils import DownloadError

    url = f'https://www.youtube.com/watch?v={ytid}'
    outtmpl = os.path.join(tmp_dir, f'{ytid}.%(ext)s')

    default_headers = {
        'User-Agent': ua or 'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Mobile Safari/537.36',
        'Referer': 'https://www.youtube.com/',
        'Accept-Language': 'en-US,en;q=0.8',
    }

    def try_once(player_client: Optional[str] = None) -> Optional[str]:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'quiet': True,
            'noprogress': True,
            'noplaylist': True,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'retries': 2,
            'http_headers': default_headers,
            'forceipv4': True,
        }
        if cookiefile:
            ydl_opts['cookiefile'] = cookiefile
        if player_client:
            # Switch player client to mitigate 403 (see yt-dlp#14680)
            ydl_opts['extractor_args'] = {
                'youtube': {
                    'player_client': [player_client],
                }
            }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except DownloadError:
            return None
        except Exception:
            return None
        if not info:
            return None
        ext = info.get('ext') or 'webm'
        path = os.path.join(tmp_dir, f'{ytid}.{ext}')
        if os.path.exists(path):
            return path
        for f in os.listdir(tmp_dir):
            if f.startswith(ytid + '.'):
                return os.path.join(tmp_dir, f)
        return None

    # Attempt sequence: default -> android -> ios -> tv
    for client in (None, 'android', 'ios', 'tv'):  # type: ignore
        path = try_once(client)
        if path:
            return path
    return None


def cut_segment_to_wav(src_path: str, dst_path: str, start: float, end: float, sr: int = 16000) -> bool:
    duration = max(0.01, float(end) - float(start))
    cmd = (
        f'ffmpeg -y -hide_banner -loglevel error -ss {start:.3f} -i "{src_path}" '
        f'-t {duration:.3f} -ar {sr} -ac 1 -vn -acodec pcm_s16le "{dst_path}"'
    )
    rc = os.system(cmd)
    return rc == 0 and os.path.exists(dst_path) and os.path.getsize(dst_path) > 0


def worker_task(item, args, mid_to_name: Dict[str, str], keep_mids: Set[str], cache_dir: str, out_dir: str) -> Dict:
    ytid, start, end, mids = item
    sel_mids = [m for m in mids if m in keep_mids]
    if not sel_mids:
        return {"status": "skip", "reason": "no_target_label", "ytid": ytid}

    os.makedirs(cache_dir, exist_ok=True)
    src_path = download_video_audio(ytid, cache_dir, ua=getattr(args, 'ua', None), cookiefile=getattr(args, 'cookies', None))
    if not src_path:
        return {"status": "fail", "reason": "download", "ytid": ytid}

    seg_key = f"{ytid}_{start:.2f}_{end:.2f}"
    seg_cache = os.path.join(cache_dir, f"{seg_key}.wav")
    if not os.path.exists(seg_cache):
        ok = cut_segment_to_wav(src_path, seg_cache, start, end, sr=args.sr)
        if not ok:
            return {"status": "fail", "reason": "ffmpeg", "ytid": ytid}

    saved = []
    for mid in sel_mids:
        name = mid_to_name.get(mid, mid)
        name_dir = os.path.join(out_dir, sanitize_name(name))
        os.makedirs(name_dir, exist_ok=True)
        dst = os.path.join(name_dir, f"{seg_key}.wav")
        if args.skip_existing and os.path.exists(dst):
            saved.append(dst)
            continue
        try:
            shutil.copyfile(seg_cache, dst)
            saved.append(dst)
        except Exception:
            pass

    return {"status": "ok", "ytid": ytid, "files": saved}


def main():
    ap = argparse.ArgumentParser(description='Download AudioSet segments to per-label folders')
    ap.add_argument('--segments_csv',
                    default=r"D:\NUS_1\CS5647_Sound_and_Music\smc-project\data\balanced_train_segments.csv",
                    help='balanced_train_segments.csv or eval_segments.csv')
    ap.add_argument('--label_csv',
                    default=r"D:\NUS_1\CS5647_Sound_and_Music\smc-project\ast\egs\audioset\data\class_labels_indices.csv",
                    help='class_labels_indices.csv')
    ap.add_argument('--out_dir', default="raw_wav", help='Output root directory')
    ap.add_argument('--keep_indices', default=None, help='Comma-separated class indices to keep')
    ap.add_argument('--keep_names', nargs='*', default=["Alarm", "Fire alarm", "Doorbell", "Knock", "Baby cry, infant cry", "Telephone bell ringing", "Vehicle horn, car horn, honking", "Civil defense siren"],
                    help='Display names to keep as a list. Example: --keep_names "Baby cry" "Doorbell" or --keep_names "[\"Baby cry\", \"Doorbell\"]"')
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--max_per_label', type=int, default=0, help='0 for no limit')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--skip_existing', default=True)
    ap.add_argument('--cache_dir', default='downloads_cache')
    # 403 mitigation options
    ap.add_argument('--ua', default=None, help='Override User-Agent for YouTube requests')
    ap.add_argument('--cookies', default=None, help='Path to a Netscape cookies.txt file for YouTube (optional)')
    args = ap.parse_args()

    ensure_deps()

    keep_names, keep_mids = parse_keep_list(args.keep_indices, args.keep_names, args.label_csv)
    keep_mid_set = set(keep_mids)
    mid_to_name, _, _ = load_label_mappings(args.label_csv)

    items = read_segments(args.segments_csv)
    items = [it for it in items if any(m in keep_mid_set for m in it[3])]

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    per_label_count: Dict[str, int] = {mid: 0 for mid in keep_mids}

    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for it in items:
            ytid, start, end, mids = it
            sel = [m for m in mids if m in keep_mid_set and (args.max_per_label <= 0 or per_label_count[m] < args.max_per_label)]
            if not sel:
                continue
            for m in sel:
                per_label_count[m] += 1
            futures.append(ex.submit(worker_task, (ytid, start, end, mids), args, mid_to_name, keep_mid_set, args.cache_dir, args.out_dir))
        for fu in as_completed(futures):
            try:
                res = fu.result()
            except Exception as e:
                res = {"status": "fail", "reason": str(e)}
            results.append(res)

    summary_path = os.path.join(args.out_dir, 'download_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'selected_labels': keep_names,
            'out_dir': os.path.abspath(args.out_dir),
            'results': results,
        }, f, ensure_ascii=False, indent=2)
    print(f'Summary written to {summary_path}')


if __name__ == '__main__':
    main()
