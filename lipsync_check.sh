#!/bin/bash
# Lip-sync check, end to end. Run as root on the GPU box:
#     bash lipsync_check.sh
# Output: /home/kg444/lipsync.tgz  (report, plots, close-up videos of the worst clips)
set -e
OUT=/root/lipsync
cd /root/digital-human
source /root/miniconda3/etc/profile.d/conda.sh

conda activate base
python lipsync_check.py tts --out $OUT

conda activate pantomatrix
source /root/.render_env
python lipsync_check.py analyze --out $OUT 2>&1 | tee $OUT/lipsync_log.txt

# face close-ups of the clips with the most issues (the analyzer prints them)
grep "python face_export.py" $OUT/lipsync_log.txt | sed 's/^ *//' | while read -r cmd; do
  eval "$cmd --out-dir $OUT/videos" || echo "  ! render failed: $cmd"
done
# plus the held vowel with vibrato, which the teacher asked about by name
python face_export.py $OUT/held_vibrato.npz --audio $OUT/wav/held_vibrato.wav \
    --label held_vibrato --out-dir $OUT/videos || true

cd /root && tar czf /home/kg444/lipsync.tgz \
    lipsync/lipsync_report.json lipsync/lipsync_log.txt lipsync/plots lipsync/wav \
    $(ls lipsync/videos/*_face.mp4 2>/dev/null)
chown kg444 /home/kg444/lipsync.tgz
echo
echo "  done -> /home/kg444/lipsync.tgz  (Download file in the SSH window)"
