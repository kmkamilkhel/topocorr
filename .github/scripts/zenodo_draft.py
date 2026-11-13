"""Create (never publish) the Zenodo record of a release as a draft, from CITATION.cff.

Usage: python .github/scripts/zenodo_draft.py --tag v0.1.0 --archive topocorr-v0.1.0.zip
       [--repo owner/name] [--dry-run]

- If CITATION.cff has a Zenodo concept DOI (``doi: 10.5281/zenodo.N``), the draft is a new version
  of that concept (made from its latest published version); otherwise it starts a new concept.
- An unpublished draft of the same concept (or, for a new concept, with the same title and
  version) is reused, so a re-run does not pile up drafts.
- The draft gets the metadata from CITATION.cff and the release source archive as its only file.
  It is left unpublished: the owner reviews and publishes it on Zenodo.

The token comes from the environment variable ZENODO_TOKEN (scopes deposit:write and
deposit:actions). ZENODO_API overrides the API root (for example the Zenodo sandbox).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

API = os.environ.get("ZENODO_API", "https://zenodo.org/api").rstrip("/")


class ZenodoError(RuntimeError):
    """An HTTP error from the Zenodo API, with its message."""


def call(method: str, url: str, token: str | None = None, body=None, data: bytes | None = None):
    """Send one API request and return the decoded JSON (or {})."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif data is not None:
        headers["Content-Type"] = "application/octet-stream"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            text = response.read()
    except urllib.error.HTTPError as err:
        raise ZenodoError(f"{method} {url}: HTTP {err.code}: {err.read().decode()[:2000]}") from err
    return json.loads(text) if text else {}


def concept_id(cff: dict) -> str | None:
    """Return the Zenodo concept record id from the ``doi`` of CITATION.cff, if any."""
    match = re.fullmatch(r"10\.5281/zenodo\.(\d+)", str(cff.get("doi", "")).strip())
    return match.group(1) if match else None


def metadata(cff: dict, tag: str, repo: str) -> dict:
    """Build the deposit metadata from CITATION.cff, as the GitHub integration wrote it."""
    creators = []
    for a in cff["authors"]:
        c = {"name": f"{a['family-names']}, {a['given-names']}"}
        if a.get("orcid"):
            c["orcid"] = str(a["orcid"]).rstrip("/").rsplit("/", 1)[-1]
        if a.get("affiliation"):
            c["affiliation"] = a["affiliation"]
        creators.append(c)
    released = cff.get("date-released")
    same_version = str(cff.get("version", "")) == tag.removeprefix("v")
    date = str(released) if released and same_version else dt.date.today().isoformat()
    return {
        "title": cff["title"],
        "upload_type": "software",
        "description": " ".join(str(cff.get("abstract", cff["title"])).split()),
        "creators": creators,
        "access_right": "open",
        "license": str(cff.get("license", "MIT")).lower(),
        "version": tag,
        "publication_date": date,
        "keywords": list(cff.get("keywords", [])),
        "related_identifiers": [
            {
                "identifier": f"https://github.com/{repo}/tree/{tag}",
                "relation": "isSupplementTo",
                "resource_type": "software",
                "scheme": "url",
            }
        ],
    }


def latest_version(concept: str) -> dict:
    """Return the latest published record of a concept (public API; follows the redirect)."""
    return call("GET", f"{API}/records/{concept}/versions/latest")


def find_draft(token: str, concept: str | None, md: dict) -> dict | None:
    """Return an unpublished draft to reuse: same concept, or same title and version."""
    drafts = call("GET", f"{API}/deposit/depositions?status=draft&size=100", token)
    for d in drafts:
        if d.get("submitted"):
            continue
        if concept and str(d.get("conceptrecid")) == concept:
            return d
        dm = d.get("metadata", {})
        if not concept and dm.get("title") == md["title"] and dm.get("version") == md["version"]:
            return d
    return None


def new_draft(token: str, concept: str | None) -> dict:
    """Create a new version of the concept, or a new concept."""
    if concept is None:
        return call("POST", f"{API}/deposit/depositions", token, body={})
    latest = latest_version(concept)
    out = call("POST", f"{API}/deposit/depositions/{latest['id']}/actions/newversion", token)
    latest_draft = out.get("links", {}).get("latest_draft")
    return call("GET", latest_draft, token) if latest_draft else out


def put_metadata(token: str, draft: dict, md: dict) -> dict:
    """Write the metadata; retry with the older licence id if the API rejects the first one."""
    url = f"{API}/deposit/depositions/{draft['id']}"
    try:
        return call("PUT", url, token, body={"metadata": md})
    except ZenodoError as err:
        if "license" not in str(err).lower() or md["license"] != "mit":
            raise
        return call("PUT", url, token, body={"metadata": {**md, "license": "mit-license"}})


def main() -> int:
    """Create or update the draft and print its link."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--cff", default="CITATION.cff", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cff = yaml.safe_load(args.cff.read_text(encoding="utf-8"))
    concept = concept_id(cff)
    md = metadata(cff, args.tag, args.repo)
    if args.dry_run:
        if concept:
            latest = latest_version(concept)
            print(f"new version of concept {concept}: from {latest['id']} ({latest['doi']})")
        else:
            print("new concept (CITATION.cff has no Zenodo concept DOI)")
        print(f"file: {args.archive.name} ({args.archive.stat().st_size} bytes)")
        print(json.dumps(md, indent=1, ensure_ascii=False))
        return 0
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not set")
        return 1
    draft = find_draft(token, concept, md)
    print("reusing draft" if draft else "creating draft", draft["id"] if draft else "")
    draft = draft or new_draft(token, concept)
    for f in call("GET", f"{API}/deposit/depositions/{draft['id']}/files", token):
        call("DELETE", f"{API}/deposit/depositions/{draft['id']}/files/{f['id']}", token)
    with args.archive.open("rb") as fh:
        call("PUT", f"{draft['links']['bucket']}/{args.archive.name}", token, data=fh.read())
    draft = put_metadata(token, draft, md)
    link = draft["links"].get("html", f"https://zenodo.org/uploads/{draft['id']}")
    doi = draft.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    kind = f"new version of concept {concept}" if concept else "new concept"
    message = f"Zenodo DRAFT ({kind}), not published: {link}  reserved DOI: {doi}"
    print(message)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Zenodo\n\n{message}\n\nReview the draft on Zenodo, then publish it.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
