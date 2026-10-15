#!/usr/bin/env python3
"""V2 Standalone CLI Entrypoint: Autonomous Streamer Viral Clipper.

Usage:
    # Auto-discover fresh streams across top US streamers:
    python main_streamer.py

    # Target specific streamer:
    python main_streamer.py --streamer ishowspeed --num-clips 3
    python main_streamer.py --streamer kai_cenat --num-clips 3

    # Process specific stream / VOD:
    python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3
"""

import sys
import argparse

# Ensure UTF-8 console output and immediate flushing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

from streamers.pipeline import run_streamer_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 AI Streamer Viral Clipper (US Streamers: Kai Cenat, Speed, Jynxzi, CaseOh, xQc)"
    )
    parser.add_argument(
        "--video-url",
        type=str,
        default="",
        help="YouTube Stream/VOD URL. If blank, auto-discovers latest streams."
    )
    parser.add_argument(
        "--streamer",
        type=str,
        default="",
        help="Target streamer filter: kai_cenat, ishowspeed, jynxzi, caseoh, xqc, adin_ross"
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=3,
        help="Number of viral clips to extract (1 to 5, default: 3)"
    )
    parser.add_argument(
        "--framing",
        type=str,
        default="auto",
        choices=["auto", "single_smooth", "split_screen", "stacked_webcam", "fullscreen_room"],
        help="Framing mode: auto, single_smooth, split_screen, stacked_webcam, fullscreen_room"
    )
    parser.add_argument(
        "--post-to-buffer",
        action="store_true",
        default=False,
        help="Post to Streamer Buffer account (strictly isolated, requires STREAMER_BUFFER_ACCESS_TOKEN)"
    )
    parser.add_argument(
        "--delay-days",
        type=int,
        default=0,
        help="Days to delay scheduled posting on Buffer (e.g. 3 or 4 for account warm-up, 0 for immediate queue)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run ID for logs and verdict directory (default: auto-generated)"
    )

    args = parser.parse_args()

    # Validate num_clips strictly (1 to 5)
    if not (1 <= args.num_clips <= 5):
        print(f"[-] Invalid --num-clips {args.num_clips}: must be between 1 and 5.")
        sys.exit(2)

    # Validate --delay-days. argparse guarantees an int, but not a sane one: a
    # negative value silently falls back to "add to queue" and an arbitrarily
    # large one schedules a post years into the future. Fail loudly instead.
    if not (0 <= args.delay_days <= 90):
        print(f"[-] Invalid --delay-days {args.delay_days}: must be between 0 and 90.")
        sys.exit(2)

    # Validate --streamer against the live roster. An unknown value (typo, odd
    # casing, or roster drift) used to fall through to discovery's auto mode and
    # clip the ENTIRE roster — publishing other creators' footage to the account
    # that was requested. Fail here, with the valid keys.
    if args.streamer.strip():
        from streamers.discovery import STREAMER_CHANNELS

        streamer_key = args.streamer.strip().lower()
        if streamer_key not in STREAMER_CHANNELS:
            print(
                f"[-] Unknown --streamer {args.streamer!r}. Valid keys: "
                + ", ".join(sorted(STREAMER_CHANNELS))
            )
            sys.exit(2)

    clips = run_streamer_pipeline(
        video_url=args.video_url.strip() or None,
        creator_name=args.streamer.strip() or None,
        num_clips=args.num_clips,
        post_to_buffer=args.post_to_buffer,
        framing_mode=args.framing,
        delay_days=args.delay_days,
        run_id=args.run_id,
    )

    if clips:
        print(f"\n[+] Successfully generated {len(clips)} streamer clips:")
        for c in clips:
            print(f"    - {c}")
        sys.exit(0)
    else:
        print("\n[-] No clips generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
