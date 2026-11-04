#!/usr/bin/env python3
"""Downloads Firefox Translations models from Mozilla's Remote Settings, the same source Firefox uses.

    fetch-models.py <dest> <languages>

<languages> is a comma-separated list of language codes (each fetched in both directions, to and
from English), or "all". A code matches its variants too: "zh" fetches zh-Hans and zh-Hant. Each direction lands in <dest>/<from>-<to>/. Files are verified against
their SHA-256 and skipped when already present; older versions in a folder are removed.

For every direction and file the newest release version wins, picking what desktop Firefox gets:
records limited to Android, Nightly or Beta are left out.
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request

RECORDS = "https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/translations-models/records"
SERVER = "https://firefox.settings.services.mozilla.com/v1/"
DESKTOP_FILTER = "env.appinfo.OS != 'Android' || env.channel != 'release'"
USER_AGENT = "reseam-translate model fetcher"


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def version_key(version):
    # "2.1" > "2.0" > "2.0a1"; pre-releases are filtered out already, but stay ordered anyway.
    parts = re.findall(r"\d+|[a-z]+", version)
    return [(int(p), 1) if p.isdigit() else (0, 0) for p in parts]


def release_records(records):
    best = {}
    for r in records:
        if r.get("filter_expression") not in ("", None, DESKTOP_FILTER):
            continue
        key = (r["fromLang"], r["toLang"], r["fileType"])
        if key not in best or version_key(r["version"]) > version_key(best[key]["version"]):
            best[key] = r
    return best


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, expected):
    folder = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=300) as response:
            h = hashlib.sha256()
            for chunk in iter(lambda: response.read(1 << 20), b""):
                h.update(chunk)
                tmp.write(chunk)
    if h.hexdigest() != expected:
        os.unlink(tmp.name)
        raise RuntimeError(f"{url}: SHA-256 mismatch")
    os.chmod(tmp.name, 0o644)
    os.replace(tmp.name, path)


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    dest, languages = sys.argv[1], sys.argv[2].strip()
    wanted = None if languages == "all" else {l.strip() for l in languages.split(",") if l.strip()}
    base = get_json(SERVER)["capabilities"]["attachments"]["base_url"]
    best = release_records(get_json(RECORDS)["data"])
    pairs = sorted({(f, t) for f, t, _ in best})
    if wanted is not None:
        def matches(code):
            return any(code == w or code.startswith(w + "-") for w in wanted)
        pairs = [(f, t) for f, t in pairs if (f == "en" and matches(t)) or (t == "en" and matches(f))]
        found = {f for f, _ in pairs} | {t for _, t in pairs}
        missing = {w for w in wanted if not any(c == w or c.startswith(w + "-") for c in found)}
        if missing:
            print(f"no models for: {', '.join(sorted(missing))}", file=sys.stderr)
    fetched = 0
    for f, t in pairs:
        folder = os.path.join(dest, f"{f}-{t}")
        os.makedirs(folder, exist_ok=True)
        keep = set()
        for kind in ("model", "lex", "vocab", "srcvocab", "trgvocab"):
            r = best.get((f, t, kind))
            if not r:
                continue
            att = r["attachment"]
            path = os.path.join(folder, att["filename"])
            keep.add(att["filename"])
            if os.path.exists(path) and sha256(path) == att["hash"]:
                continue
            download(base + att["location"], path, att["hash"])
            fetched += att["size"]
        for name in os.listdir(folder):
            # Older versions go; the engine's packed cache (model.reseam) stays, it checks itself.
            if name not in keep and name != "model.reseam":
                os.unlink(os.path.join(folder, name))
        print(f"{f}-{t} ok", file=sys.stderr)
    print(f"{len(pairs)} directions in {dest}, {fetched / 1e6:.0f} MB downloaded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
