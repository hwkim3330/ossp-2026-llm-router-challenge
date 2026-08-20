#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
#
# Finish the OSSP 2026 submission once the two things only a human can do are
# done: authorising a registry push, and uploading the demo video.
#
#   gh auth refresh -h github.com -s write:packages
#   YOUTUBE_URL=https://youtu.be/... ./finish_submission.sh
#
# It pushes the already-built image, records the digest in
# submission-ossp-skt.json, commits that as its own commit (the challenge
# requires the JSON commit to come *after* the code commit it points at), then
# fills the report's two placeholders and rebuilds the PDF and the upload zip.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

CODE_SHA="$(git rev-parse HEAD)"
IMAGE="ghcr.io/hwkim3330/ossp-router"
DOCS="$HOME/Documents"
: "${YOUTUBE_URL:?set YOUTUBE_URL to the uploaded demo video link}"

echo "== code commit: $CODE_SHA"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; commit or stash before building the image" >&2
  exit 1
fi

echo "== logging in to ghcr.io"
gh auth token | docker login ghcr.io -u hwkim3330 --password-stdin

echo "== building and pushing linux/arm64 from $CODE_SHA"
docker buildx build --platform linux/arm64 --push \
  --provenance=false --sbom=false \
  --file router/Dockerfile --tag "$IMAGE:submission" router/

DIGEST="$(docker buildx imagetools inspect "$IMAGE:submission" \
          | awk '/^Digest:/ {print $2; exit}')"
echo "== image digest: $DIGEST"

python3 - "$CODE_SHA" "$IMAGE@$DIGEST" <<'PY'
import json, sys
sha, image = sys.argv[1], sys.argv[2]
json.dump({
    "schema_version": 1,
    "challenge_id": "ossp-2026-llm-router-challenge",
    "repository_url": "https://github.com/hwkim3330/ossp-2026-llm-router-challenge",
    "commit_sha": sha,
    "image_digest": image,
    "primary_license": "Apache-2.0",
}, open("submission-ossp-skt.json", "w"), indent=2, ensure_ascii=False)
open("submission-ossp-skt.json", "a").write("\n")
PY

python3 tools/validate_technical_submission.py

git add submission-ossp-skt.json
git commit -q -m "Record the submission image digest built from ${CODE_SHA:0:7}"
git push origin main
SNAP_SHA="$(git rev-parse HEAD)"
SNAPSHOT="https://github.com/hwkim3330/ossp-2026-llm-router-challenge/tree/$SNAP_SHA"
echo "== 프로젝트 등록 URL: $SNAPSHOT"

echo "== filling the report placeholders"
python3 - "$SNAPSHOT" "$YOUTUBE_URL" <<'PY'
import sys
from pathlib import Path
from docx import Document

snapshot, video = sys.argv[1], sys.argv[2]
path = Path.home() / "Documents/2026 오픈소스 개발자대회 결과보고서_168(메타몽).docx"
doc = Document(str(path))
hits = 0
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    for token, value in (("{{SNAPSHOT_URL}}", snapshot),
                                         ("{{YOUTUBE_URL}}", video)):
                        if token in run.text:
                            run.text = run.text.replace(token, value)
                            hits += 1
assert hits >= 2, f"placeholders not found (replaced {hits})"
doc.save(str(path))
print("filled", hits, "placeholders")
PY

cd "$DOCS"
libreoffice --headless --convert-to pdf \
  "2026 오픈소스 개발자대회 결과보고서_168(메타몽).docx" >/dev/null
rm -f "제출_168_메타몽.zip"
zip -j "제출_168_메타몽.zip" \
  "2026 오픈소스 개발자대회 결과보고서_168(메타몽).docx" \
  "2026 오픈소스 개발자대회 결과보고서_168(메타몽).pdf"

echo
echo "== done"
echo "  프로젝트 등록 URL : $SNAPSHOT"
echo "  시연영상          : $YOUTUBE_URL"
echo "  이미지            : $IMAGE@$DIGEST"
echo "  업로드 파일       : $DOCS/제출_168_메타몽.zip"
