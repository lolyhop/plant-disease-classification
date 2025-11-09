#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/raw"
ZIP_PATH="${DATA_DIR}/new-plant-diseases-dataset.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/vipoooool/new-plant-diseases-dataset"

VALID_RATIO=0.5

mkdir -p "${DATA_DIR}"

echo "[1] Downloading dataset..."
curl -L -o "${ZIP_PATH}" "${URL}"

echo "[2] Unzipping dataset..."
unzip -q -o "${ZIP_PATH}" -d "${DATA_DIR}"

echo "[3] Flattening directory structure..."
for split in train valid test; do
  find "${DATA_DIR}" -type d -name "${split}" | while read -r nested_dir; do
    mv -n "${nested_dir}" "${DATA_DIR}/" 2>/dev/null || true
  done
done

find "${DATA_DIR}" -type d -empty -not -path "${DATA_DIR}" -delete
rm -f "${ZIP_PATH}"

TRAIN_DIR="${DATA_DIR}/train"
VALID_DIR="${DATA_DIR}/valid"
TEST_DIR="${DATA_DIR}/test"

if [ ! -d "${TRAIN_DIR}" ] || [ ! -d "${VALID_DIR}" ]; then
  echo "Error: train or valid folder missing. Dataset structure unexpected."
  exit 1
fi

echo "[4] Removing old test set if present..."
rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"

echo "[5] Splitting valid into valid + test (stratified per class)..."
for class_dir in "${VALID_DIR}"/*; do
  if [ -d "${class_dir}" ]; then
    class_name="$(basename "${class_dir}")"
    mkdir -p "${TEST_DIR}/${class_name}"

    mapfile -t files < <(find "${class_dir}" -type f | shuf)
    total=${#files[@]}
    split_point=$(printf "%.0f" "$(echo "$total * $VALID_RATIO" | bc)")

    for i in "${!files[@]}"; do
      file="${files[$i]}"
      if [ "$i" -ge "$split_point" ]; then
        mv "${file}" "${TEST_DIR}/${class_name}/"
      fi
    done
  fi
done

echo "[6] Dataset ready:"
echo " - train: $(find "${TRAIN_DIR}" -type f | wc -l) files"
echo " - valid: $(find "${VALID_DIR}" -type f | wc -l) files"
echo " - test:  $(find "${TEST_DIR}" -type f | wc -l) files"

echo "Done."
