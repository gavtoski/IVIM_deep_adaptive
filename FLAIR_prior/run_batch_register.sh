#!/bin/bash
# run_batch_register.sh
# Calls Register_lesion_FLAIR2IVIM.sh for a list of subjects.
# Skips subjects if result_folder/<ID> already exists.

set -euo pipefail

# Path to your registration script (assumes it's in the same directory as this file)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTER_SH="${SCRIPT_DIR}/Register_lesion_FLAIR2IVIM.sh"

if [[ ! -x "${REGISTER_SH}" ]]; then
  echo "[ERROR] ${REGISTER_SH} not found or not executable."
  echo "        Make it executable with: chmod +x Register_lesion_FLAIR2IVIM.sh"
  exit 1
fi

subjects=(NC145 NC144 NC142 NC135 NC130 NC129 NC128)

for sid in "${subjects[@]}"; do
  if [[ -d "result_folder/${sid}" ]]; then
    echo "[SKIP] result_folder/${sid} exists — skipping ${sid}"
    continue
  fi

  echo "[RUN] Processing ${sid} ..."
  bash "${REGISTER_SH}" "${sid}"
  echo "[DONE] ${sid}"
done

echo "[ALL DONE]"
