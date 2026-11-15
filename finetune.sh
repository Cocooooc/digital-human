#!/bin/bash
# Fine-tune EMAGE on another BEAT2 speaker. As root, from anywhere:
#   bash finetune.sh check            codebook check on the 8 diagnosed speakers (~3 min)
#   bash finetune.sh prep  6          download + preprocess speaker 6 (~10 min)
#   bash finetune.sh train 6          fine-tune (~1 h on a T4 for 3000 steps)
#   bash finetune.sh eval  6          before/after numbers + 3-way video
#   bash finetune.sh all   6          prep + train + eval
set -e
STAGE=$1; SPK=${2:-0}
cd /root/digital-human
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pantomatrix
source /root/.render_env
mkdir -p /root/ft
run() { python /root/digital-human/finetune_emage.py "$@" 2>&1 | tee -a /root/ft/log_$1.txt; }
case $STAGE in
  check) run check ;;
  prep)  run prep $SPK ;;
  train) run train $SPK ;;
  eval)  run eval $SPK
         D=/root/ft/eval_speaker$SPK
         for s in data original finetuned; do
           python face_export.py $D/$s.npz --audio $D/audio.wav --out-dir $D --keep-root --view upper --size 640 --label "$s" || true
         done
         ffmpeg -y -loglevel error -i $D/data_face.mp4 -i $D/original_face.mp4 -i $D/finetuned_face.mp4 \
           -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" -map 0:a -c:v libx264 -crf 20 \
           -pix_fmt yuv420p -c:a aac $D/compare_3way.mp4 || true
         cd /root && tar czf /home/kg444/ft_eval.tgz ft/eval_speaker$SPK.md ft/eval_speaker$SPK.json ft/log_*.txt \
           ft/codebook_check.json ft/speaker$SPK/loss_history.json ft/eval_speaker$SPK/compare_3way.mp4 2>/dev/null || true
         chown kg444 /home/kg444/ft_eval.tgz; ls -lh /home/kg444/ft_eval.tgz ;;
  all)   bash $0 prep $SPK && bash $0 train $SPK && bash $0 eval $SPK ;;
  *) echo "usage: bash finetune.sh check|prep|train|eval|all SPEAKER" ;;
esac
