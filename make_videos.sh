#!/bin/bash
# make_videos.sh — compile flow field and dashboard frames into mp4 + gif
# Run after train.py completes.
#
# Usage:
#   bash make_videos.sh

set -euo pipefail

RUN="Re100_uniformU_1xhidden"
FLOW_DIR="images/flow_${RUN}"
DASH_DIR="images/dashboard_${RUN}"
mkdir -p videos

echo "Flow frames  : $(ls $FLOW_DIR/epoch=*.png | wc -l)"
echo "Dash frames  : $(ls $DASH_DIR/epoch=*.png | wc -l)"
echo

# ── Flow field mp4 ─────────────────────────────────────────────────────────
echo "Building flow mp4..."
ls "$FLOW_DIR"/epoch=*.png | sort | \
  while read f; do echo "file '$(pwd)/$f'"; echo "duration 0.004"; done > /tmp/flow_frames.txt
ffmpeg -y -f concat -safe 0 -i /tmp/flow_frames.txt \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 -crf 28 -preset slow -pix_fmt yuv420p \
  videos/flow_${RUN}.mp4
echo "  → videos/flow_${RUN}.mp4"

# ── Flow field gif ─────────────────────────────────────────────────────────
echo "Building flow gif..."
ffmpeg -y -i videos/flow_${RUN}.mp4 \
  -vf "fps=50,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  -loop -1 videos/flow_${RUN}.gif
echo "  → videos/flow_${RUN}.gif"

# ── Dashboard mp4 ──────────────────────────────────────────────────────────
echo "Building dashboard mp4..."
ls "$DASH_DIR"/epoch=*.png | sort | \
  while read f; do echo "file '$(pwd)/$f'"; echo "duration 0.04"; done > /tmp/dash_frames.txt
ffmpeg -y -f concat -safe 0 -i /tmp/dash_frames.txt \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 -crf 28 -preset slow -pix_fmt yuv420p \
  videos/dashboard_${RUN}.mp4
echo "  → videos/dashboard_${RUN}.mp4"

# ── Dashboard gif ──────────────────────────────────────────────────────────
echo "Building dashboard gif..."
ffmpeg -y -i videos/dashboard_${RUN}.mp4 \
  -vf "fps=15,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  -loop -1 videos/dashboard_${RUN}.gif
echo "  → videos/dashboard_${RUN}.gif"

echo
echo "Done. Videos saved to videos/"
