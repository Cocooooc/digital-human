#!/bin/bash
# Remaining problems on top of the fine-tune + one table for all five. As root:
#     bash final_eval.sh 4
# Output: $DL/final_eval.tgz  (five_problems.md/json, blink_model.npz, compare_4way.mp4)
set -e
# where results are left for download: the GCP login user's home, or /root/results elsewhere
if id kg444 >/dev/null 2>&1; then DL=/home/kg444; else DL=/root/results; fi; mkdir -p $DL
SPK=${1:-4}
cd /root/digital-human
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pantomatrix
source /root/.render_env
# huggingface.co is unreachable from mainland China: use the mirror there.
if ! curl -s -m 6 -o /dev/null https://huggingface.co; then export HF_ENDPOINT=https://hf-mirror.com; echo "  using $HF_ENDPOINT"; fi
D=/root/ft/final_speaker$SPK
mkdir -p $D
PYTHONUNBUFFERED=1 python -u final_eval.py --speaker $SPK 2>&1 | tee $D/log.txt
for v in v1_lawrence v2_original v3_finetuned v4_full; do
  python face_export.py $D/$v.npz --audio $D/audio.wav --out-dir $D --keep-root --view upper --size 540 --label "$v" || true
done
ffmpeg -y -loglevel error -i $D/v1_lawrence_face.mp4 -i $D/v2_original_face.mp4 -i $D/v3_finetuned_face.mp4 -i $D/v4_full_face.mp4 \
  -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" -map "[v]" -map 0:a -c:v libx264 -crf 21 -pix_fmt yuv420p \
  -c:a aac $D/compare_4way.mp4 || true
# face close-ups too: blinks and eyes are invisible at body scale
for v in v1_lawrence v3_finetuned v4_full; do
  python face_export.py $D/$v.npz --audio $D/audio.wav --out-dir $D/face --keep-root --size 400 --label "$v" || true
done
ffmpeg -y -loglevel error -i $D/face/v1_lawrence_face.mp4 -i $D/face/v3_finetuned_face.mp4 -i $D/face/v4_full_face.mp4 \
  -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" -map 0:a -c:v libx264 -crf 21 -pix_fmt yuv420p \
  -c:a aac $D/compare_face_3way.mp4 || true
cd /root && tar czf $DL/final_eval.tgz ft/final_speaker$SPK/five_problems.md ft/final_speaker$SPK/five_problems.json \
  ft/final_speaker$SPK/log.txt ft/final_speaker$SPK/blink_model.npz ft/final_speaker$SPK/compare_4way.mp4 \
  ft/final_speaker$SPK/compare_face_3way.mp4 2>/dev/null || true
chown kg444 $DL/final_eval.tgz 2>/dev/null || true
ls -lh $DL/final_eval.tgz
echo "  done -> download $DL/final_eval.tgz (open a NEW SSH window first)"
