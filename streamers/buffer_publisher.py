"""Dedicated Streamer Buffer Publisher (Pipeline V2).

Strictly isolated from Podcast Buffer credentials:
- Uses STREAMER_BUFFER_ACCESS_TOKEN and STREAMER_BUFFER_CHANNEL_IDS
- Formats TikTok / Shorts metadata with authentic streamer hashtags (2-4 tags)
- Uploads assets to GitHub Release for direct public video URLs required by Buffer
- Supports Buffer Queue as well as specific scheduled due dates (e.g. in 3-4 days)
"""

import os
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from core.github_uploader import upload_clip_to_github_release
from streamers.viral_detector import StreamerClipCandidate

BUFFER_GRAPHQL_ENDPOINT = "https://api.buffer.com"


class StreamerBufferPublisher:
    def __init__(
        self,
        access_token: Optional[str] = None,
        channel_ids: Optional[List[str]] = None
    ):
        self.token = access_token or os.getenv("STREAMER_BUFFER_ACCESS_TOKEN", "").strip()
        env_channel_ids = [
            c.strip() for c in os.getenv("STREAMER_BUFFER_CHANNEL_IDS", "").split(",") if c.strip()
        ]
        self.channel_ids = channel_ids or env_channel_ids
        self.last_error: Optional[str] = None

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def get_channels(self) -> List[Dict[str, Any]]:
        """Queries connected social channels from Buffer GraphQL."""
        if not self.token:
            self.last_error = "STREAMER_BUFFER_ACCESS_TOKEN is missing"
            return []

        org_query = """
        query GetOrgs {
          account {
            organizations {
              id
              name
            }
          }
        }
        """
        try:
            res = requests.post(
                BUFFER_GRAPHQL_ENDPOINT,
                headers=self.headers,
                json={"query": org_query},
                timeout=15
            )
            if res.status_code != 200:
                self.last_error = f"Buffer orgs query failed: {res.status_code}"
                return []

            orgs = res.json().get("data", {}).get("account", {}).get("organizations", [])
            channels = []
            for org in orgs:
                org_id = org.get("id")
                if not org_id:
                    continue
                ch_query = """query GetChannels($organizationId: OrganizationId!) {
                  channels(input: { organizationId: $organizationId }) {
                    id
                    name
                    service
                  }
                }"""
                ch_res = requests.post(
                    BUFFER_GRAPHQL_ENDPOINT,
                    headers=self.headers,
                    json={"query": ch_query, "variables": {"organizationId": org_id}},
                    timeout=15
                )
                if ch_res.status_code == 200:
                    for ch in ch_res.json().get("data", {}).get("channels", []):
                        ch["organizationId"] = org_id
                        ch["organizationName"] = org.get("name")
                        channels.append(ch)
            return channels
        except Exception as e:
            self.last_error = str(e)
            return []

    def schedule_clip(
        self,
        clip_path: Path,
        candidate: StreamerClipCandidate,
        thumb_path: Optional[Path] = None,
        target_channel_ids: Optional[List[str]] = None,
        due_at: Optional[str] = None,
        delay_days: Optional[int] = None,
        target_hour_utc: int = 18  # 18:00 UTC = 2:00 PM EST (Peak US scroll time)
    ) -> List[Dict[str, Any]]:
        """Uploads clip to GitHub release and schedules it to Buffer social channels.

        Args:
            clip_path: Path to rendered MP4 video
            candidate: StreamerClipCandidate with title, caption, hashtags
            thumb_path: Optional path to thumbnail image
            target_channel_ids: Specific channel IDs, or defaults to self.channel_ids
            due_at: Specific ISO 8601 timestamp string (e.g. '2026-10-15T18:00:00Z')
            delay_days: Relative days to delay (e.g. 3 or 4 days from now)
            target_hour_utc: Target posting hour in UTC (default 18:00 UTC)
        """
        if not self.token:
            print("[-] Cannot post to Buffer: STREAMER_BUFFER_ACCESS_TOKEN is missing.")
            return []

        # 1. Upload video to GitHub Release for direct public hosting
        print("[*] Uploading streamer clip to GitHub release for Buffer CDN access...")
        video_url = upload_clip_to_github_release(clip_path)
        if not video_url:
            print(f"[-] Could not get public video URL for {clip_path.name}. Cannot post to Buffer.")
            return []

        print(f"[+] Public video URL: {video_url}")

        _thumbnail_url = None
        if thumb_path and thumb_path.exists():
            _thumbnail_url = upload_clip_to_github_release(thumb_path)
            if _thumbnail_url:
                print(f"[+] Public thumbnail URL: {_thumbnail_url}")

        # 2. Resolve channels — fail CLOSED
        connected = self.get_channels()
        if not connected:
            print("[-] No channels found in Streamer Buffer account.")
            return []

        channel_map = {ch["id"]: ch for ch in connected}
        chosen_ids = list(target_channel_ids or self.channel_ids or [])

        # A misconfigured STREAMER_BUFFER_CHANNEL_IDS must never degrade into a
        # broadcast. The previous fallback silently fanned one clip out to every
        # connected channel — and, when the account had no TikTok channel, to
        # every connected service. A config typo could therefore publish to
        # accounts nobody selected. Refuse instead, and name what to fix.
        unknown_ids = [cid for cid in chosen_ids if cid not in channel_map]
        if unknown_ids:
            self.last_error = f"unknown channel ids: {', '.join(unknown_ids)}"
            print(
                "[-] Refusing to publish: STREAMER_BUFFER_CHANNEL_IDS lists "
                f"{len(unknown_ids)} channel id(s) absent from this Buffer "
                f"account: {', '.join(unknown_ids)}"
            )
            return []

        if not chosen_ids:
            self.last_error = "no target channel resolved"
            print(
                "[-] Refusing to publish: no target channel resolved. Set "
                "STREAMER_BUFFER_CHANNEL_IDS (comma-separated Buffer channel ids) "
                "or pass target_channel_ids explicitly. Available channels: "
                + ", ".join(
                    f"{ch['id']} ({ch.get('service', '?')})" for ch in connected
                )
            )
            return []

        # 3. Resolve scheduled due date
        scheduled_due_at = due_at
        if not scheduled_due_at and delay_days is not None and delay_days > 0:
            target_date = datetime.now(timezone.utc) + timedelta(days=delay_days)
            scheduled_time = target_date.replace(hour=target_hour_utc, minute=0, second=0, microsecond=0)
            scheduled_due_at = scheduled_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 4. Prepare caption strictly preserving streamer hashtags (2 to 4 tags)
        tags_str = " ".join(candidate.hashtags[:4])
        caption_text = f"{candidate.title}\n\n{candidate.social_caption}\n\n{tags_str}".strip()

        mutation = """
        mutation CreateStreamerPost($input: CreatePostInput!) {
          createPost(input: $input) {
            ... on PostActionSuccess {
              post {
                id
                status
                dueAt
              }
            }
            ... on MutationError {
              message
            }
          }
        }
        """

        results = []
        for ch_id in chosen_ids:
            ch_info = channel_map.get(ch_id, {})
            service = ch_info.get("service", "tiktok")
            ch_name = ch_info.get("name", ch_id)

            metadata: Dict[str, Any] = {}
            if service == "tiktok":
                metadata = {
                    "tiktok": {
                        "title": candidate.title[:90],
                        "isAiGenerated": False
                    }
                }
            elif service == "youtube":
                yt_title = candidate.title
                if "#shorts" not in yt_title.lower():
                    yt_title = f"{yt_title} #shorts"
                metadata = {
                    "youtube": {
                        "title": yt_title[:95],
                        "categoryId": "20",  # Gaming / Entertainment
                        "madeForKids": False,
                        "privacy": "public"
                    }
                }
            elif service == "instagram":
                metadata = {
                    "instagram": {
                        "type": "reel",
                        "shouldShareToFeed": True
                    }
                }

            post_input: Dict[str, Any] = {
                "channelId": ch_id,
                "text": caption_text,
                "schedulingType": "automatic",
                "mode": "customScheduled" if scheduled_due_at else "addToQueue",
                "needsApproval": False,
                "assets": [
                    {
                        "video": {
                            "url": video_url
                        }
                    }
                ]
            }

            if scheduled_due_at:
                post_input["dueAt"] = scheduled_due_at

            if metadata:
                post_input["metadata"] = metadata

            print(f"[*] Dispatching streamer clip to Buffer channel: {ch_name} ({service}) | Mode: {post_input['mode']}...")
            if scheduled_due_at:
                print(f"    -> Scheduled Due Date: {scheduled_due_at}")

            # Retry logic — ONLY for failures that provably did NOT create a post.
            #
            # Buffer's createPost has no idempotency key, so it is NOT idempotent:
            # a retry after an ambiguous failure (timeout, connection reset, or a
            # response we failed to parse after the request was already delivered)
            # can create a SECOND post for the same clip on the same channel. The
            # old code retried on *any* exception, which is exactly that case.
            #
            #   retryable : Buffer explicitly rejected the media URL (CDN
            #               propagation delay) — it tells us no post was created.
            #   terminal  : any other API-level rejection.
            #   ambiguous : transport error with no usable response. Never
            #               retried; recorded as "unknown" for a human to
            #               reconcile rather than silently duplicated.
            max_retries = 3
            retry_delays = [10, 25, 45]
            propagation_keywords = [
                "video could not be read",
                "unable to read video",
                "invalid video url",
            ]

            for attempt in range(max_retries + 1):
                if attempt > 0:
                    print(f"[*] Retrying Buffer dispatch ({attempt}/{max_retries}) after {retry_delays[attempt-1]}s...")
                    time.sleep(retry_delays[attempt-1])

                try:
                    res = requests.post(
                        BUFFER_GRAPHQL_ENDPOINT,
                        headers=self.headers,
                        json={"query": mutation, "variables": {"input": post_input}},
                        timeout=20
                    )
                except Exception as exc:
                    # The request may or may not have reached Buffer. Retrying
                    # here risks a duplicate post, so stop and flag it instead.
                    print(
                        f"[-] Ambiguous transport error sending to Buffer {ch_name}: {exc}. "
                        f"Not retrying (a retry could create a duplicate post) — "
                        f"reconcile manually."
                    )
                    results.append({
                        "channel_id": ch_id,
                        "channel_name": ch_name,
                        "service": service,
                        "error": str(exc),
                        "status": "unknown",
                        "retry_safe": False,
                        # Keep the URL so a human can reconcile against Buffer's
                        # queue without re-deriving it.
                        "video_url": video_url,
                    })
                    break

                try:
                    res_data = res.json()
                except Exception:
                    body = (res.text or "")[:200]
                    print(f"[-] Buffer returned an unparseable body on {ch_name}: {body!r}")
                    results.append({
                        "channel_id": ch_id,
                        "channel_name": ch_name,
                        "service": service,
                        "error": f"unparseable response: {body}",
                        "status": "unknown",
                        "retry_safe": False,
                    })
                    break

                create_data = res_data.get("data", {}).get("createPost", {}) or {}

                if "post" in create_data:
                    post_obj = create_data["post"] or {}
                    post_id = post_obj.get("id")
                    post_due = post_obj.get("dueAt")
                    print(f"[+] Successfully scheduled on Buffer! Channel: {ch_name} (ID: {post_id}, Due: {post_due})")
                    results.append({
                        "channel_id": ch_id,
                        "channel_name": ch_name,
                        "service": service,
                        "post_id": post_id,
                        "due_at": post_due,
                        "video_url": video_url,
                        "status": "scheduled"
                    })
                    break

                error_msg = (
                    create_data.get("message")
                    or (res_data.get("errors") or [{}])[0].get("message")
                    or res.text
                )
                if any(kw in str(error_msg).lower() for kw in propagation_keywords) and attempt < max_retries:
                    print(f"[-] Buffer CDN propagation wait: {error_msg}")
                    continue

                print(f"[-] Buffer rejected post on {ch_name}: {error_msg}")
                results.append({
                    "channel_id": ch_id,
                    "channel_name": ch_name,
                    "service": service,
                    "error": error_msg,
                    "status": "rejected"
                })
                break

        return results

    def get_posts(self, channel_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Queries posts from specified or all connected channels."""
        if not self.token:
            self.last_error = "STREAMER_BUFFER_ACCESS_TOKEN is missing"
            return []

        channels = self.get_channels()
        if not channels:
            return []

        target_ids = channel_ids or self.channel_ids or [ch["id"] for ch in channels]
        all_posts = []

        query = """
        query ListPosts($organizationId: OrganizationId!, $channelIds: [ChannelId!]) {
          posts(input: {organizationId: $organizationId, filter: {channelIds: $channelIds}}) {
            edges {
              node {
                id
                status
                text
                dueAt
                channelId
                assets {
                  ... on VideoAsset {
                    source
                  }
                }
              }
            }
          }
        }
        """

        for ch in channels:
            ch_id = ch["id"]
            if ch_id not in target_ids:
                continue
            org_id = ch.get("organizationId", "default_org")

            try:
                res = requests.post(
                    BUFFER_GRAPHQL_ENDPOINT,
                    headers=self.headers,
                    json={
                        "query": query,
                        "variables": {
                            "organizationId": org_id,
                            "channelIds": [ch_id],
                        },
                    },
                    timeout=20,
                )
                if res.status_code != 200:
                    print(f"[-] Buffer posts query failed for channel {ch.get('name', ch_id)}: status {res.status_code}")
                    continue
                body = res.json()
                edges = body.get("data", {}).get("posts", {}).get("edges", []) or []
                for edge in edges:
                    node = edge.get("node", {})
                    if node:
                        node["channel_name"] = ch.get("name")
                        node["service"] = ch.get("service")
                        all_posts.append(node)
            except Exception as e:
                print(f"[-] Error fetching posts for channel {ch.get('name', ch_id)}: {e}")

        return all_posts

    def delete_post(self, post_id: str) -> bool:
        """Deletes a post by ID from Buffer via GraphQL mutation."""
        if not self.token:
            self.last_error = "STREAMER_BUFFER_ACCESS_TOKEN is missing"
            return False

        # Attempt deletePost first (standard Buffer GraphQL schema)
        del_mutation = """
        mutation DeletePost($input: DeletePostInput!) {
          deletePost(input: $input) {
            __typename
            ... on DeletePostSuccess {
              id
            }
            ... on MutationError {
              message
            }
          }
        }
        """
        try:
            res = requests.post(
                BUFFER_GRAPHQL_ENDPOINT,
                headers=self.headers,
                json={"query": del_mutation, "variables": {"input": {"id": post_id}}},
                timeout=15,
            )
            data = res.json() if res.status_code == 200 else {}
            del_result = data.get("data", {}).get("deletePost", {}) or {}
            typename = del_result.get("__typename")
            if typename == "DeletePostSuccess" or del_result.get("id") == post_id:
                return True

            errors = data.get("errors") or []
            if any("deletePost" in err.get("message", "") for err in errors):
                # Fallback to destroyPost
                destroy_mutation = """
                mutation DestroyPost($input: DestroyPostInput!) {
                  destroyPost(input: $input) {
                    __typename
                    ... on PostActionSuccess {
                      post {
                        id
                      }
                    }
                    ... on MutationError {
                      message
                    }
                  }
                }
                """
                res2 = requests.post(
                    BUFFER_GRAPHQL_ENDPOINT,
                    headers=self.headers,
                    json={"query": destroy_mutation, "variables": {"input": {"id": post_id}}},
                    timeout=15,
                )
                data2 = res2.json() if res2.status_code == 200 else {}
                post_id_res = (
                    data2.get("data", {})
                    .get("destroyPost", {})
                    .get("post", {})
                    .get("id")
                )
                if post_id_res == post_id:
                    return True

            err = del_result.get("message") or (errors[0].get("message") if errors else res.text[:200])
            self.last_error = f"Failed to delete post {post_id}: {err}"
            return False
        except Exception as e:
            self.last_error = str(e)
            return False

    def delete_matching_posts(self, pattern: str) -> List[str]:
        """Finds and deletes any queued post matching pattern in text or video source."""
        pattern_lower = pattern.lower().strip()
        if not pattern_lower:
            return []

        posts = self.get_posts()
        deleted_ids = []
        for post in posts:
            post_id = post.get("id")
            text = (post.get("text") or "").lower()
            sources = []
            for asset in post.get("assets", []):
                s = asset.get("source") or ""
                if s:
                    sources.append(s.lower())

            matches = (pattern_lower in text) or any(pattern_lower in s for s in sources)
            if matches and post_id:
                print(f"[*] Found matching post to delete in Buffer: ID={post_id} | Status={post.get('status')} | Text={post.get('text', '')[:60]}...")
                if self.delete_post(post_id):
                    print(f"[+] Successfully deleted post {post_id} from Buffer.")
                    deleted_ids.append(post_id)
                else:
                    print(f"[-] Failed to delete post {post_id}: {self.last_error}")

        return deleted_ids
