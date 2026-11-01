"""Resolve an online video (e.g. YouTube) to a direct stream URL that OpenCV can read.

Nothing is downloaded or saved: yt-dlp only looks up the stream address, and
``VideoFileSource`` then reads frames from the network as they play. Needs
``pip install -e ".[youtube]"``. Only use videos you are allowed to use; check the
video's licence and the site's terms.

YouTube sometimes answers automated lookups with "Sign in to confirm you're not a bot"
(it depends on your network address and how many lookups it has seen). When that happens,
let yt-dlp borrow a logged-in browser's cookies, once, via an environment variable:

    FACETRACK_YT_COOKIES_BROWSER=chrome        # or firefox, edge, brave, ...
    FACETRACK_YT_COOKIES_FILE=cookies.txt      # alternative: a Netscape-format cookie file

Use your own account's cookies on your own machine only, and never commit or share them.
"""

from __future__ import annotations

import os


def ytdlp_options(max_height: int = 720) -> dict:
    opts: dict = {
        # One H.264 stream (OpenCV decodes it everywhere); audio is not needed.
        "format": f"bv*[vcodec^=avc1][height<={max_height}]/b[height<={max_height}]/b",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if browser := os.environ.get("FACETRACK_YT_COOKIES_BROWSER"):
        opts["cookiesfrombrowser"] = (browser,)
    if cookie_file := os.environ.get("FACETRACK_YT_COOKIES_FILE"):
        opts["cookiefile"] = cookie_file
    return opts


def resolve_stream(url: str, max_height: int = 720) -> tuple[str, dict]:
    """Return (direct stream URL, yt-dlp info dict: title, channel, duration, width, height, fps)."""
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError('URL input needs: pip install -e ".[youtube]"') from e
    try:
        with yt_dlp.YoutubeDL(ytdlp_options(max_height)) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "confirm you" in msg and "bot" in msg:
            raise RuntimeError(
                "YouTube asked for a sign-in before it would give out this video (it does this to automated "
                "lookups from some networks). Set FACETRACK_YT_COOKIES_BROWSER=chrome (or firefox, edge) so "
                "yt-dlp can use your browser's login, or try again later; see src/facetrack/download.py."
            ) from e
        raise RuntimeError(msg.replace("ERROR: ", "", 1)) from e
    stream = info.get("url") or next((f["url"] for f in info.get("requested_formats", []) if f.get("vcodec") != "none"), None)
    if not stream:
        raise RuntimeError("could not find a playable video stream for this URL")
    return stream, info


def resolve_stream_url(url: str, max_height: int = 720) -> str:
    stream, info = resolve_stream(url, max_height)
    print(f"streaming: {info.get('title')!r} ({info.get('width')}x{info.get('height')}, {info.get('fps')} fps)")
    return stream
