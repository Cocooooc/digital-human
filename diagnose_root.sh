#!/bin/bash
# Root-cause diagnosis for the five chosen problems. As root:
#     bash diagnose_root.sh
# Output: /home/kg444/diag.tgz
set -e
cd /root/digital-human
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pantomatrix
source /root/.render_env
mkdir -p /root/diag
python diagnose_root.py --out /root/diag 2>&1 | tee /root/diag/diag_log.txt || true
cd /root && tar czf /home/kg444/diag.tgz diag
chown kg444 /home/kg444/diag.tgz
ls -lh /home/kg444/diag.tgz
echo "  done -> download /home/kg444/diag.tgz (open a NEW SSH window first)"
