#!/bin/bash
# Chinese clips for the robot + lip-sync before/after, end to end. As root:
#     bash make_chinese_clips.sh
# Output: /home/kg444/final.tgz
#   zh_clips/for_robot/<clip>/   npz + npy (body, expression, jaw, eyes) per clip
#   zh_clips/videos/             <clip>_face.mp4 (processed) and <clip>_raw_face.mp4
#   lipsync/                     before/after report + plots (English test set)
set -e
cd /root/digital-human
source /root/miniconda3/etc/profile.d/conda.sh
OUT=/root/zh_clips

# 1. Mandarin voice. Usage:
#      bash make_chinese_clips.sh          Microsoft neural voice (edge-tts, natural; needs internet)
#      bash make_chinese_clips.sh kokoro   offline Kokoro (flatter)
#      bash make_chinese_clips.sh own      your own recordings, already uploaded as
#                                          /root/zh_clips/wav/zh_01_intro.(wav|m4a|mp3) etc.
ENGINE=${1:-edge}
mkdir -p $OUT/wav
if [ "$ENGINE" = "own" ]; then
  for f in $OUT/wav/*.m4a $OUT/wav/*.mp3 $OUT/wav/*.MP3 $OUT/wav/*.M4A; do
    [ -e "$f" ] || continue
    ffmpeg -y -loglevel error -i "$f" -ac 1 -ar 24000 "${f%.*}.wav" && echo "  converted $(basename "$f")"
  done
  ls -l $OUT/wav/zh_0*.wav || { echo "  ! put your recordings in $OUT/wav first"; exit 1; }
else
  conda activate base
  if [ "$ENGINE" = "edge" ]; then
    pip install -q -U edge-tts
    python make_chinese_clips.py tts --out $OUT --engine edge \
      || { echo "  ! edge-tts failed (network?) — falling back to Kokoro"; ENGINE=kokoro; }
  fi
  if [ "$ENGINE" = "kokoro" ]; then
    python -c "import misaki.zh" 2>/dev/null || pip install -q "misaki[zh]"
    python make_chinese_clips.py tts --out $OUT --engine kokoro
  fi
fi

# 2. motion (+ mouth fix, idle, gaze), raw kept for comparison
conda activate pantomatrix
source /root/.render_env
python make_chinese_clips.py motion --out $OUT 2>&1 | tee $OUT/motion_log.txt

# 3. robot export (body + face npy) and videos
for f in $OUT/zh_0*_raw.npz; do
  c=$(basename $f _raw.npz)
  python export_for_robot.py $OUT/$c.npz --out-dir $OUT/for_robot/$c --name $c
  python face_export.py $OUT/$c.npz --audio $OUT/${c}_track.wav --out-dir $OUT/videos \
      --side-by-side --keep-root --label "$c  (mouth fix + idle + gaze)"
  python face_export.py $OUT/${c}_raw.npz --audio $OUT/${c}_raw_track.wav --out-dir $OUT/videos \
      --side-by-side --keep-root --label "$c  (EMAGE raw)"
done

# 4. lip-sync check, now with before/after
python lipsync_check.py analyze --out /root/lipsync --post 2>&1 | tee /root/lipsync/lipsync_post_log.txt
for c in held_vowel normal_3 pauses; do
  python face_export.py /root/lipsync/${c}_fixed.npz --audio /root/lipsync/wav/$c.wav \
      --out-dir /root/lipsync/videos --label "$c  after mouth fix" || true
  python face_export.py /root/lipsync/$c.npz --audio /root/lipsync/wav/$c.wav \
      --out-dir /root/lipsync/videos --label "$c  EMAGE raw" || true
done

cd /root && tar czf /home/kg444/final.tgz \
    zh_clips/for_robot zh_clips/videos/*_face.mp4 zh_clips/clips.json zh_clips/motion_log.txt zh_clips/wav \
    lipsync/lipsync_report.json lipsync/lipsync_post_log.txt lipsync/plots lipsync/videos/*_face.mp4
chown kg444 /home/kg444/final.tgz
ls -lh /home/kg444/final.tgz
echo "  done -> download /home/kg444/final.tgz (open a NEW SSH window first)"
