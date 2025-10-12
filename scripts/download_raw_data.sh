#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/raw"
ZIP_PATH="${DATA_DIR}/new-plant-diseases-dataset.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/vipoooool/new-plant-diseases-dataset"

mkdir -p "${DATA_DIR}"

# echo "Downloading dataset..."
curl -L -o "${ZIP_PATH}" "${URL}"

# echo "Unzipping..."
unzip -q -o "${ZIP_PATH}" -d "${DATA_DIR}"

# Flatten nested folders
for split in train valid test; do
  find "${DATA_DIR}" -type d -name "${split}" | while read -r nested_dir; do
    mv -n "${nested_dir}" "${DATA_DIR}/" 2>/dev/null || true
  done
done

# Remove any empty directories left after moving
find "${DATA_DIR}" -type d -empty -not -path "${DATA_DIR}" -delete

# Clean up archive
rm -f "${ZIP_PATH}"

echo "Dataset ready at ${DATA_DIR} with the following splits:"
for s in train valid test; do
  if [ -d "${DATA_DIR}/${s}" ]; then
    echo " - ${s}: $(find "${DATA_DIR}/${s}" -type f | wc -l) files"
  fi
done
