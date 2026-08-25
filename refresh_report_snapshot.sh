#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
#
# Re-point the report at the current repository snapshot, without touching the
# image or the code commit it was built from.
#
# Needed because the report's `프로젝트 등록 URL` is a fixed tree/<sha> snapshot,
# so any commit made after finish_submission.sh ran -- a README, a doc fix -- is
# invisible at the URL the judges open. The image digest and the JSON's
# commit_sha stay exactly as they were: only the snapshot moves, and every commit
# in this repository carries submission-ossp-skt.json in its tree, which is what
# the rules require the snapshot commit to contain.
#
#   YOUTUBE_URL=https://youtu.be/... ./refresh_report_snapshot.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
DOCS="$HOME/Documents"
: "${YOUTUBE_URL:?set YOUTUBE_URL to the uploaded demo video link}"

[ -z "$(git status --porcelain)" ] || { echo "working tree dirty; commit first" >&2; exit 1; }
git diff --quiet origin/main..HEAD || { echo "HEAD not pushed; push first" >&2; exit 1; }

SNAP_SHA="$(git rev-parse HEAD)"
CODE_SHA="$(python3 -c 'import json;print(json.load(open("submission-ossp-skt.json"))["commit_sha"])')"
SNAPSHOT="https://github.com/hwkim3330/ossp-2026-llm-router-challenge/tree/$SNAP_SHA"
git cat-file -e "$SNAP_SHA:submission-ossp-skt.json" \
  || { echo "snapshot commit does not contain the JSON" >&2; exit 1; }
echo "== 프로젝트 등록 URL: $SNAPSHOT"
echo "== 이미지 빌드 커밋 (변경 없음): $CODE_SHA"

python3 "$HOME/ossp-report/fill_report.py"
python3 - "$SNAPSHOT" "$YOUTUBE_URL" "$CODE_SHA" <<'PY'
import sys
from pathlib import Path
from docx import Document

snapshot, video, code_sha = sys.argv[1], sys.argv[2], sys.argv[3]
path = Path.home() / "Documents/2026 오픈소스 개발자대회 결과보고서_168(메타몽).docx"
doc = Document(str(path))
hits = 0
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    for token, value in (("{{SNAPSHOT_URL}}", snapshot),
                                         ("{{YOUTUBE_URL}}", video),
                                         ("{{CODE_SHA}}", code_sha)):
                        if token in run.text:
                            run.text = run.text.replace(token, value)
                            hits += 1
assert hits >= 3, f"placeholders not found (replaced {hits})"
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
echo "== rebuilt $DOCS/제출_168_메타몽.zip"
