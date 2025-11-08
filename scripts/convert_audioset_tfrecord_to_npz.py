import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert AudioSet v1 TFRecords to per-sample NPZ files (embeddings + labels)")
    parser.add_argument('--in_dir', default=r"D:\NUS_1\CS5647_Sound_and_Music\smc-project\data\features\audioset_v1_embeddings\bal_train\tf_record", help='Directory containing .tfrecord files')
    parser.add_argument('--out_dir', default=r"D:\NUS_1\CS5647_Sound_and_Music\smc-project\data\features\audioset_v1_embeddings\bal_train\npz", help='Output directory for .npz files')
    parser.add_argument('--prefix', default='', help='Optional filename prefix for outputs')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit of samples to convert')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from tfrecord import tfrecord_loader
        from tfrecord.tools import tfrecord2idx
    except Exception as e:
        raise SystemExit(
            'This script requires the "tfrecord" package (pip install tfrecord). ' 
            'Network may be restricted; consider offline install.\n'
            f'Import error: {e}'
        )

    # AudioSet v1 embeddings are stored as a SequenceExample:
    # - context features: labels, start_time_seconds, end_time_seconds, video_id
    # - sequence feature: audio_embedding (bytes per frame)
    description = {
        'labels': 'int',
        'start_time_seconds': 'float',
        'end_time_seconds': 'float',
        'video_id': 'byte',
    }
    sequence_description = {
        'audio_embedding': 'byte',
    }

    tfrecords = [os.path.join(args.in_dir, f) for f in os.listdir(args.in_dir) if f.endswith('.tfrecord')]
    tfrecords.sort()
    total = 0
    dupe_guard = set()
    for tfrec in tfrecords:
        idx_path = tfrec + '.index'
        if not os.path.exists(idx_path):
            tfrecord2idx.create_index(tfrec, idx_path)

        for i, sample in enumerate(tfrecord_loader(
            tfrec,
            idx_path,
            description=description,
            sequence_description=sequence_description,
        )):
            # tfrecord_loader with sequence_description returns (context_dict, sequence_dict)
            if isinstance(sample, tuple) and len(sample) == 2:
                ctx, seq = sample
            else:
                ctx, seq = sample, sample

            # audio_embedding is typically a list of bytes (one per frame).
            # Try common variants if key differs.
            emb = None
            for k in ('audio_embedding', 'embedding', 'audio_embedding/bytes'):
                if isinstance(seq, dict) and (k in seq):
                    emb = seq[k]
                    break
                if isinstance(ctx, dict) and (k in ctx):
                    emb = ctx[k]
                    break
            if emb is None:
                # If still missing, skip this sample
                continue

            if isinstance(emb, (list, tuple)):
                frames = [np.frombuffer(b, dtype=np.uint8) for b in emb]
                x = np.stack(frames).astype(np.float32) / 255.0  # [T,128]
            else:
                # assume contiguous frames in a single bytes object
                arr = np.frombuffer(emb, dtype=np.uint8).astype(np.float32)
                if arr.size % 128 != 0:
                    # cannot reshape; skip sample
                    continue
                x = (arr / 255.0).reshape(-1, 128)

            # labels from context
            labels = []
            if isinstance(ctx, dict):
                labels = ctx.get('labels', [])
            try:
                y = np.array(labels, dtype=np.int64).reshape(-1)
            except Exception:
                # if labels not numeric, skip
                continue

            # video id
            vid = b''
            if isinstance(ctx, dict):
                vid = ctx.get('video_id', b'')
            if isinstance(vid, (bytes, bytearray)):
                vid_str = vid.decode('utf-8', errors='ignore')
            elif isinstance(vid, (list, tuple)) and len(vid) > 0 and isinstance(vid[0], (bytes, bytearray)):
                vid_str = vid[0].decode('utf-8', errors='ignore')
            else:
                vid_str = str(vid)

            # times
            start = 0.0
            end = 0.0
            if isinstance(ctx, dict):
                s = ctx.get('start_time_seconds', 0.0)
                e = ctx.get('end_time_seconds', 0.0)
                if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0:
                    s = s[0]
                if isinstance(e, (list, tuple, np.ndarray)) and len(e) > 0:
                    e = e[0]
                start = float(s)
                end = float(e)
            key = f"{vid_str}_{int(round(start*1000))}_{int(round(end*1000))}"
            if key in dupe_guard:
                # ensure uniqueness by appending counter
                key = f"{key}_{i}"
            dupe_guard.add(key)

            out_name = f"{args.prefix}{key}.npz" if args.prefix else f"{key}.npz"
            out_path = os.path.join(args.out_dir, out_name)
            np.savez_compressed(out_path, embedding=x, labels=y, video_id=vid_str, start=start, end=end)

            total += 1
            print(total)
            if args.limit is not None and total >= args.limit:
                print(f"Converted {total} samples; reached limit")
                return

    print(f"Converted total {total} samples to {args.out_dir}")


if __name__ == '__main__':
    main()
