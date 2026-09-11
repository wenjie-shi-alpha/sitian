#!/usr/bin/env bash
set -euo pipefail

source_dir="/root/sitian_open_evidence/raw/nwp/ifs"
target_dir="/mnt/c/sitian_open_evidence_raw/nwp/ifs"
manifest="/root/sitian_open_evidence/manifests/open_evidence_v1.json"
log="/root/sitian_open_evidence/logs/ifs.log"

if [[ -L "${source_dir}" ]]; then
  printf 'IFS raw path is already a symlink: %s\n' "${source_dir}"
  exit 0
fi
if [[ ! -d "${source_dir}" || ! -d /mnt/c ]]; then
  printf 'source or /mnt/c is unavailable\n' >&2
  exit 2
fi
available_kib="$(df --output=avail -k /mnt/c | tail -1 | tr -d ' ')"
if [[ "${available_kib}" -lt 600000000 ]]; then
  printf '/mnt/c has less than the required 600 GB safety floor\n' >&2
  exit 3
fi

tmux kill-session -t sitian_hybrid_ifs 2>/dev/null || true
while pgrep -f 'fetch_open_nwp.py.*--source ifs' >/dev/null; do
  sleep 1
done

mkdir -p "${target_dir}"
# Files are removed from the ext4 source only after rsync has completed each
# copy successfully.  The destination is the recoverable copy.
rsync -a --remove-source-files --info=progress2 "${source_dir}/" "${target_dir}/"
if find "${source_dir}" -type f -print -quit | grep -q .; then
  printf 'source still contains files after rsync; refusing to replace it\n' >&2
  exit 4
fi
find "${source_dir}" -depth -type d -empty -delete
if [[ -e "${source_dir}" ]]; then
  printf 'source directory is not empty; refusing to replace it\n' >&2
  exit 5
fi
ln -s "${target_dir}" "${source_dir}"

tmux new-session -d -s sitian_hybrid_ifs \
  "cd /root/sitian && PYTHONPATH=src python3 -u scripts/fetch_open_nwp.py --manifest '${manifest}' --root /root/sitian_open_evidence --source ifs --product both --workers 16 > '${log}' 2>&1"
printf 'IFS migration complete: %s -> %s; download restarted\n' "${source_dir}" "${target_dir}"
