#!/bin/bash
# Website greeting + before/after comparison, end to end. Run as root on the GPU box:
#     bash make_greeting.sh                       # default text
#     bash make_greeting.sh "Hi! I'm ..."         # your own text
# Output (copied to /home/kg444 for Download file):
#   greeting.mp4          1080x1080 waist-up, the finished greeting
#   greeting_compare.mp4  1920x960, raw EMAGE (left) vs with our motion layer (right)
#   greeting_for_robot/   the greeting's motion as npz/npy for the robot
set -e
OUT=/root/greeting
cd /root/digital-human
source /root/miniconda3/etc/profile.d/conda.sh

conda activate base
if [ -n "${1:-}" ]; then
  python make_greeting.py tts --out $OUT --text "$1"
else
  python make_greeting.py tts --out $OUT
fi

conda activate pantomatrix
source /root/.render_env
python make_greeting.py motion --out $OUT

# the greeting itself
python face_export.py $OUT/greeting.npz --audio $OUT/greeting.wav \
    --keep-root --view upper --label "" --size 1080

# comparison: render both halves at 960, then put them side by side
python face_export.py $OUT/greeting_raw.npz --audio $OUT/greeting.wav --out-dir $OUT/cmp \
    --keep-root --view upper --size 960 --label "Raw EMAGE output"
python face_export.py $OUT/greeting.npz --audio $OUT/greeting.wav --out-dir $OUT/cmp \
    --keep-root --view upper --size 960 --label "+ idle, gaze, softer gestures"
ffmpeg -y -loglevel error -i $OUT/cmp/greeting_raw_face.mp4 -i $OUT/cmp/greeting_face.mp4 \
    -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 1:a \
    -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart \
    $OUT/greeting_compare.mp4

# motion data for the robot
python export_for_robot.py $OUT/greeting.npz --out-dir $OUT/greeting_for_robot --name greeting \
    || { echo "  ! robot export failed — videos are still fine"; mkdir -p $OUT/greeting_for_robot; }

cp $OUT/greeting_face.mp4 /home/kg444/greeting.mp4
cp $OUT/greeting_compare.mp4 /home/kg444/greeting_compare.mp4
rm -rf /home/kg444/greeting_for_robot && cp -r $OUT/greeting_for_robot /home/kg444/
chown -R kg444 /home/kg444/greeting.mp4 /home/kg444/greeting_compare.mp4 /home/kg444/greeting_for_robot
cd /home/kg444 && tar czf greeting_for_robot.tgz greeting_for_robot && chown kg444 greeting_for_robot.tgz
echo
echo "  done. Download file in the SSH window:"
echo "    /home/kg444/greeting.mp4"
echo "    /home/kg444/greeting_compare.mp4"
echo "    /home/kg444/greeting_for_robot.tgz"
