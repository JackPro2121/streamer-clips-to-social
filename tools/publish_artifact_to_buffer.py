#!/usr/bin/env python3
"""CLI Tool to publish existing generated & gate-passed streamer clips directly to Buffer.

Useful when clips have already been rendered, audited, and verified in a previous run,
allowing immediate queueing without having to re-download or re-render source video.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure UTF-8 console output and immediate flushing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

from streamers.buffer_publisher import StreamerBufferPublisher
from streamers.viral_detector import StreamerClipCandidate


def publish_clips_from_dir(directory: Path, delay_days: int = 0) -> int:
    publisher = StreamerBufferPublisher()
    if not publisher.token:
        print("[-] STREAMER_BUFFER_ACCESS_TOKEN is missing. Cannot publish.")
        return 1

    mp4_files = sorted(directory.glob("*.mp4"))
    if not mp4_files:
        # Check subdirectories
        mp4_files = sorted(directory.rglob("*.mp4"))

    if not mp4_files:
        print(f"[-] No .mp4 files found in {directory}")
        return 1

    print(f"[+] Found {len(mp4_files)} clip(s) to publish in {directory}")

    success_count = 0
    for clip_path in mp4_files:
        # Check for associated edl.json
        edl_path = clip_path.with_suffix(".edl.json")
        thumb_path = None
        raw_title = clip_path.stem
        clean_title = raw_title
        hashtags = ["#streamer", "#clips", "#viral", "#shorts"]
        caption = "BRO IS ACTUALLY UNHINGED 💀 Wait till the end..."

        if clean_title.startswith("clip_"):
            # Strip clip_1_ prefix and trailing hash if present
            parts = clean_title.split("_")
            if len(parts) >= 3:
                # clip, 1, title..., hash
                inner = parts[2:]
                if len(inner[-1]) == 6:  # hex hash
                    inner = inner[:-1]
                clean_title = " ".join(inner)

        if edl_path.exists():
            try:
                edl_data = json.loads(edl_path.read_text(encoding="utf-8"))
                clean_title = edl_data.get("title", clean_title)
                creator = edl_data.get("creator", "")
                if creator:
                    creator_tag = f"#{creator.lower().replace(' ', '')}"
                    if creator_tag not in hashtags:
                        hashtags.insert(0, creator_tag)
                social_caption = edl_data.get("social_caption")
                if social_caption:
                    caption = social_caption
                else:
                    c_name = creator.upper() if creator else "BRO"
                    caption = f"{c_name} IS ACTUALLY UNHINGED 💀 Wait till the end..."
                # Check thumbnail path
                raw_thumb = edl_data.get("thumbnail_path")
                if raw_thumb:
                    t_cand = Path(raw_thumb)
                    if t_cand.exists():
                        thumb_path = t_cand
                    else:
                        local_t = clip_path.parent / t_cand.name
                        if local_t.exists():
                            thumb_path = local_t
            except Exception as e:
                print(f"[!] Warning reading {edl_path.name}: {e}")

        # If thumb_path not found, search in clip directory
        if not thumb_path:
            for cand in clip_path.parent.glob(f"*{clip_path.stem}*thumb*.jpg"):
                thumb_path = cand
                break
            if not thumb_path:
                for cand in clip_path.parent.glob("*.jpg"):
                    thumb_path = cand
                    break

        candidate = StreamerClipCandidate(
            title=clean_title[:45],
            start_time=0.0,
            end_time=38.0,
            duration=38.0,
            viral_score=95,
            hook_reason="Verified viral streamer moment",
            social_caption=caption,
            hashtags=hashtags,
            streamer_energy="hype",
        )

        print(f"\n[*] Publishing {clip_path.name} ('{clean_title}')...")
        results = publisher.schedule_clip(
            clip_path=clip_path,
            candidate=candidate,
            thumb_path=thumb_path,
            delay_days=delay_days,
        )

        if results:
            print(f"[+] Successfully queued {clip_path.name} to {len(results)} channel(s):")
            for res in results:
                cid = res.get("channel_id", "?")
                svc = res.get("service", "?")
                pid = res.get("post_id", "?")
                print(f"    → channel={cid} ({svc}) post_id={pid}")
            success_count += 1
        else:
            print(f"[-] Failed to schedule {clip_path.name}. Error: {publisher.last_error}")

    return 0 if success_count > 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish already generated streamer clips to Buffer")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing clip .mp4 files")
    parser.add_argument("--delay-days", type=int, default=0, help="Days to delay posting (0 for queue)")

    args = parser.parse_args()
    dir_path = Path(args.dir)
    if not dir_path.exists():
        print(f"[-] Directory {dir_path} does not exist.")
        sys.exit(1)

    exit_code = publish_clips_from_dir(dir_path, delay_days=args.delay_days)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
