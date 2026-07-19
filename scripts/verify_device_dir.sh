#!/usr/bin/env bash
#
# verify_device_dir.sh — verify (and repair) a devicectl-pushed directory.
#
# devicectl `device copy to` has been observed to SILENTLY DROP files on large
# trees (2026-07-20: 219 of 2030 files missing from a 3.8 GB Cactus bundle, exit 0).
# This compares the local listing with `devicectl device info files` and re-pushes
# the difference one file at a time.
#
#   scripts/verify_device_dir.sh <local-dir> <device-subdirectory> [udid] [bundle-id]
#
# Flat directories only (the Cactus CQ bundles are flat). Exit 0 = device matches.
set -uo pipefail

LOCAL="${1:?local dir}"; DEVDIR="${2:?device subdirectory (Documents/…)}"
UDID="${3:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"
BUNDLE_ID="${4:-com.daisukemajima.llmbench}"

tmp_local="$(mktemp)"; tmp_dev="$(mktemp)"; trap 'rm -f "$tmp_local" "$tmp_dev"' EXIT
ls "$LOCAL" | sort > "$tmp_local"
xcrun devicectl device info files --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --subdirectory "$DEVDIR" 2>/dev/null \
  | awk '{print $1}' | sort > "$tmp_dev"

missing="$(comm -23 "$tmp_local" "$tmp_dev")"
if [ -z "$missing" ]; then
  echo "OK: device matches local ($(wc -l < "$tmp_local" | tr -d ' ') files)"
  exit 0
fi

n="$(echo "$missing" | wc -l | tr -d ' ')"
echo "device is missing $n files — re-pushing individually…"
fails=0
while IFS= read -r f; do
  xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" --source "$LOCAL/$f" --destination "$DEVDIR/$f" \
    > /dev/null 2>&1 || { fails=$((fails+1)); echo "FAIL $f"; }
done <<< "$missing"
echo "re-pushed $((n-fails))/$n"
[ "$fails" -eq 0 ]
