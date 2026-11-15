#!/bin/bash
# AI screening, stage 1: BEAT2 data vs EMAGE (argmax / sampled). As root:
#     bash audit_emage.sh
# Output: /home/kg444/audit.tgz  (audit_report.md/json, plots/, faces/)
set -e
cd /root/digital-human
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pantomatrix
source /root/.render_env
mkdir -p /root/audit
python audit_emage.py --out /root/audit 2>&1 | tee /root/audit/audit_log.txt || true
cd /root && tar czf /home/kg444/audit.tgz audit/audit_report.md audit/audit_report.json audit/audit_log.txt audit/plots audit/faces
chown kg444 /home/kg444/audit.tgz
ls -lh /home/kg444/audit.tgz
echo "  done -> download /home/kg444/audit.tgz (open a NEW SSH window first)"
