#!/usr/bin/env bash
# Run full pipeline for questions.jsonl:
# 1) Download video from HuggingFace (JakeTian/M3-web-video)
# 2) Cut video into 30-second clips
# 3) Download intermediate outputs (faces, voices)
# 4) Build memory graph
# 5) Answer questions for this video
# 6) Delete video, clips, intermediate_outputs, and .pkl to free storage
#
# Requirements: pip install huggingface_hub, ffmpeg, and m3-agent dependencies
#
# Usage:
#   ./run_questions_pipeline.sh                    # Process all videos in questions.jsonl
#   ./run_questions_pipeline.sh --limit 2         # Process first 2 videos
#   ./run_questions_pipeline.sh HLDPA3FTUJ4      # Process specific video(s)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HF_VIDEO_DATASET="JakeTian/M3-web-video"
HF_INTERMEDIATE_DATASET="JakeTian/m3-intermediate-output"
QUESTIONS_FILE="data/annotations/web.json"
RESULTS_FILE="data/results/web.json"
TOKEN_FILE="data/results/token_consumption.json"
INTERVAL_SECONDS=30
MAX_PARALLEL=5
VIDEOS_DIR="data/videos/web"
CLIPS_DIR="data/clips/web"
INTERMEDIATE_DIR="data/intermediate_outputs/web"
MEM_DIR="data/memory_graphs/web"

mkdir -p "$VIDEOS_DIR" "$CLIPS_DIR" "$INTERMEDIATE_DIR" "$MEM_DIR" data/results

download_video() {
  local video_id="$1"
  local output_path="${VIDEOS_DIR}/${video_id}.mp4"
  if [[ -f "$output_path" && -s "$output_path" ]]; then
    echo "  Video already exists, skipping download: $output_path"
    return 0
  fi
  echo "  Downloading ${video_id}.mp4 from HuggingFace..."
  if python3 -c "
from huggingface_hub import hf_hub_download
import shutil
import os
try:
    path = hf_hub_download(
        repo_id='$HF_VIDEO_DATASET',
        filename='${video_id}.mp4',
        repo_type='dataset',
        local_dir='$VIDEOS_DIR',
        local_dir_use_symlinks=False
    )
    # hf_hub_download returns path to file; may be in subdir
    if os.path.exists(path):
        dst = '$output_path'
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.move(path, dst)
        os._exit(0)
except Exception as e:
    print(f'Download error: {e}')
    os._exit(1)
" 2>/dev/null; then
    if [[ -f "$output_path" && -s "$output_path" ]]; then
      echo "  ✓ Downloaded $output_path"
      return 0
    fi
  fi
  # Fallback: huggingface-cli
  if command -v huggingface-cli &>/dev/null; then
    if huggingface-cli download "$HF_VIDEO_DATASET" "${video_id}.mp4" \
      --repo-type dataset \
      --local-dir "$VIDEOS_DIR" \
      --local-dir-use-symlinks False 2>/dev/null; then
      if [[ -f "${VIDEOS_DIR}/${video_id}.mp4" && -s "${VIDEOS_DIR}/${video_id}.mp4" ]]; then
        echo "  ✓ Downloaded $output_path"
        return 0
      fi
      if [[ -f "${VIDEOS_DIR}/${video_id}/${video_id}.mp4" ]]; then
        mv "${VIDEOS_DIR}/${video_id}/${video_id}.mp4" "$output_path"
        rm -rf "${VIDEOS_DIR}/${video_id}"
        echo "  ✓ Downloaded $output_path"
        return 0
      fi
    fi
  fi
  echo "  ✗ Download failed for ${video_id}. Install: pip install huggingface_hub"
  return 1
}

download_intermediate_outputs() {
  local video_id="$1"
  local output_dir="${INTERMEDIATE_DIR}/${video_id}"
  if [[ -d "$output_dir" && -n "$(ls -A "$output_dir" 2>/dev/null)" ]]; then
    echo "  Intermediate outputs already exist, skipping download: $output_dir"
    return 0
  fi
  echo "  Downloading intermediate outputs for ${video_id} from HuggingFace..."
  if python3 -c "
from huggingface_hub import snapshot_download
import os
import sys
try:
    snapshot_download(
        repo_id='$HF_INTERMEDIATE_DATASET',
        repo_type='dataset',
        allow_patterns='${video_id}/**',
        local_dir='$INTERMEDIATE_DIR',
        local_dir_use_symlinks=False
    )
    if os.path.isdir('$output_dir') and os.listdir('$output_dir'):
        sys.exit(0)
except Exception as e:
    print(f'Download error: {e}')
    sys.exit(1)
sys.exit(1)
" 2>/dev/null; then
    echo "  ✓ Downloaded intermediate outputs to $output_dir"
    return 0
  fi
  # Fallback: huggingface-cli
  if command -v huggingface-cli &>/dev/null; then
    if huggingface-cli download "$HF_INTERMEDIATE_DATASET" \
      --repo-type dataset \
      --include "${video_id}/**" \
      --local-dir "$INTERMEDIATE_DIR" \
      --local-dir-use-symlinks False 2>/dev/null; then
      if [[ -d "$output_dir" && -n "$(ls -A "$output_dir" 2>/dev/null)" ]]; then
        echo "  ✓ Downloaded intermediate outputs to $output_dir"
        return 0
      fi
    fi
  fi
  echo "  ✗ Intermediate outputs download failed for ${video_id}. Install: pip install huggingface_hub"
  return 1
}

cut_video_into_clips() {
  local video_id="$1"
  local input="${VIDEOS_DIR}/${video_id}.mp4"
  local clip_base="${CLIPS_DIR}/${video_id}"
  mkdir -p "$clip_base"
  if [[ ! -f "$input" || ! -s "$input" ]]; then
    echo "  ✗ Video file not found: $input"
    return 1
  fi
  local duration
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input" 2>/dev/null || echo "0")
  local duration_seconds
  duration_seconds=$(echo "$duration" | awk '{print int($1)}')
  if [[ "$duration_seconds" -le 0 ]]; then
    echo "  ✗ Could not get video duration"
    return 1
  fi
  local segments=$((duration_seconds / INTERVAL_SECONDS + 1))
  echo "  Cutting into $segments segment(s)..."
  for ((i = 0; i < segments; i++)); do
    local start=$((i * INTERVAL_SECONDS))
    local output="${clip_base}/${i}.mp4"
    local remaining=$((duration_seconds - start))
    if [[ "$remaining" -le 0 ]]; then
      continue
    fi
    local segment_duration=$INTERVAL_SECONDS
    if [[ "$remaining" -lt "$INTERVAL_SECONDS" ]]; then
      if [[ "$remaining" -lt 1 ]]; then
        continue
      fi
      segment_duration=$remaining
    fi
    # Re-encode to avoid corrupted clips from keyframe cuts
    ffmpeg -y -ss "$start" -i "$input" -t "$segment_duration" \
      -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
      -c:a aac -movflags +faststart \
      "$output" 2>/dev/null && true
  done
  echo "  ✓ Clips saved to $clip_base"
  return 0
}

cleanup_video() {
  local video_id="$1"
  echo "  Cleaning up video, clips, intermediate_outputs, and memory graph for ${video_id}..."
  rm -f "${VIDEOS_DIR}/${video_id}.mp4"
  rm -rf "${CLIPS_DIR}/${video_id}"
  rm -rf "${INTERMEDIATE_DIR}/${video_id}"
  rm -f "${MEM_DIR}/${video_id}.pkl"
  echo "  ✓ Cleanup complete for ${video_id}"
}

process_video() {
  local video_id="$1"
  local token_file="data/results/token_consumption_${video_id}.json"
  echo ""
  echo "============================================================"
  echo "Processing video: ${video_id}"
  echo "============================================================"

  # Step 1: Download
  if ! download_video "$video_id"; then
    return 1
  fi

  # Step 2: Cut into clips
  if ! cut_video_into_clips "$video_id"; then
    rm -f "${VIDEOS_DIR}/${video_id}.mp4"
    return 1
  fi

  # Step 3: Create data.jsonl for this video
  local data_file="data/data_${video_id}.jsonl"
  local clip_path="${CLIPS_DIR}/${video_id}"
  local mem_path="${MEM_DIR}/${video_id}.pkl"
  local intermediate_path="${INTERMEDIATE_DIR}/${video_id}"
  echo "{\"id\": \"${video_id}\", \"clip_path\": \"${clip_path}\", \"mem_path\": \"${mem_path}\", \"intermediate_outputs\": \"${intermediate_path}\"}" > "$data_file"

  # Step 4: Download intermediate outputs
  if ! download_intermediate_outputs "$video_id"; then
    echo "  ✗ Intermediate outputs download failed for ${video_id}"
    rm -f "$data_file"
    cleanup_video "$video_id"
    return 1
  fi

  # Step 5: Build memory graph
  echo "  Building memory graph..."
  if ! python -m m3_agent.memorization_memory_graphs --data_file "$data_file" --token_file "$token_file"; then
    echo "  ✗ Memory graph failed for ${video_id}"
    rm -f "$data_file"
    cleanup_video "$video_id"
    return 1
  fi
  rm -f "$data_file"

  # Step 6: Create filtered questions file and run control
  local questions_filtered="data/annotations/questions_${video_id}.json"
  python3 -c "
import json
with open('$QUESTIONS_FILE', 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered = {k: v for k, v in data.items() if k == '$video_id'}
with open('$questions_filtered', 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=4)
    f.write('\\n')
" || true

  local question_count
  question_count=$(python3 -c "
import json
with open('$questions_filtered', 'r', encoding='utf-8') as f:
    data = json.load(f)
count = 0
for _, v in data.items():
    count += len(v.get('qa_list', []))
print(count)
" 2>/dev/null || echo "0")
  if [[ "$question_count" -eq 0 ]]; then
    echo "  No questions found for ${video_id}, skipping QA"
    cleanup_video "$video_id"
    return 0
  fi

  echo "  Answering ${question_count} question(s)..."
  if python -m m3_agent.control \
    --data_file "$questions_filtered" \
    --mem_path_template "${MEM_DIR}/{video_id}.pkl" \
    --token_file "$token_file"; then
    echo "  ✓ Answered questions for ${video_id}"
  else
    echo "  ✗ QA failed for ${video_id}"
  fi

  rm -f "$questions_filtered"

  # Step 7: Cleanup
  cleanup_video "$video_id"
  return 0
}

# Parse arguments
LIMIT=""
VIDEO_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS] [VIDEO_IDS...]"
      echo ""
      echo "Process videos from questions.jsonl: download, build memory, answer, cleanup."
      echo ""
      echo "Options:"
      echo "  --limit N     Process first N unique videos from questions.jsonl"
      echo "  VIDEO_IDS    Specific video IDs to process (e.g. HLDPA3FTUJ4)"
      echo ""
      echo "Examples:"
      echo "  $0                           # Process all videos"
      echo "  $0 --limit 2                 # Process first 2 videos"
      echo "  $0 HLDPA3FTUJ4 NiKqiIYwVAg  # Process specific videos"
      exit 0
      ;;
    *)
      VIDEO_ARGS+=("$1")
      shift
      ;;
  esac
done

# Get video list
if [[ ${#VIDEO_ARGS[@]} -gt 0 ]]; then
  VIDEOS=("${VIDEO_ARGS[@]}")
else
  echo "Extracting unique video IDs from $QUESTIONS_FILE..."
  VIDEOS=($(python3 -c "
import json
with open('$QUESTIONS_FILE', 'r', encoding='utf-8') as f:
    data = json.load(f)
for vid in data.keys():
    print(vid)
" | tr '\n' ' '))
fi

if [[ ${#VIDEOS[@]} -eq 0 ]]; then
  echo "✗ No videos to process"
  exit 1
fi

# Apply limit
if [[ -n "$LIMIT" ]]; then
  VIDEOS=("${VIDEOS[@]:0:$LIMIT}")
  echo "Processing first $LIMIT video(s) (limit applied)"
fi

echo "Processing ${#VIDEOS[@]} video(s): ${VIDEOS[*]}"

# Process each video (parallel)
FAILED=0
for video_id in "${VIDEOS[@]}"; do
  while [[ $(jobs -pr | wc -l | tr -d ' ') -ge $MAX_PARALLEL ]]; do
    if ! wait -n; then
      ((FAILED++)) || true
    fi
  done
  process_video "$video_id" &
done

while [[ $(jobs -pr | wc -l | tr -d ' ') -gt 0 ]]; do
  if ! wait -n; then
    ((FAILED++)) || true
  fi
done

# Merge per-video results in original order
: > "$RESULTS_FILE"
for video_id in "${VIDEOS[@]}"; do
  if [[ -f "data/results/questions_${video_id}.jsonl" ]]; then
    cat "data/results/questions_${video_id}.jsonl" >> "$RESULTS_FILE"
    rm -f "data/results/questions_${video_id}.jsonl"
  fi
done

# Merge per-video token files
python3 - <<'PY'
import json
from pathlib import Path

root = Path("data/results")
out_path = root / "token_consumption.json"
result = {
    "memory": {"total": 0, "generation": 0, "embedding": 0, "by_video": {}},
    "control": {"total": 0, "by_video": {}},
}

for token_file in sorted(root.glob("token_consumption_*.json")):
    try:
        data = json.loads(token_file.read_text(encoding="utf-8"))
    except Exception:
        continue
    mem = data.get("memory", {})
    ctl = data.get("control", {})
    result["memory"]["total"] += mem.get("total", 0) or 0
    result["memory"]["generation"] += mem.get("generation", 0) or 0
    result["memory"]["embedding"] += mem.get("embedding", 0) or 0
    result["control"]["total"] += ctl.get("total", 0) or 0
    result["memory"]["by_video"].update(mem.get("by_video", {}) or {})
    result["control"]["by_video"].update(ctl.get("by_video", {}) or {})

out_path.write_text(json.dumps(result, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
for token_file in root.glob("token_consumption_*.json"):
    token_file.unlink(missing_ok=True)
PY

echo ""
echo "============================================================"
echo "Pipeline complete. Processed ${#VIDEOS[@]} video(s), $FAILED failed."
echo "Results written to $RESULTS_FILE"
echo "Token consumption saved to $TOKEN_FILE"
echo "============================================================"
