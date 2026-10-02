#!/usr/bin/env bash
# Grant raw-packet capture access to every built Starspin FPGA/channel variant.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_APPS="$ROOT/build/apps"
DRY_RUN=false
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --build-dir)
      [[ $# -ge 2 ]] || { echo '--build-dir requires the directory containing binaries' >&2; exit 1; }
      BUILD_APPS=$2; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--build-dir PATH]"
      echo 'Applies cap_net_raw=ep to built starspin_1_* through starspin_4_* binaries, including experimental variants.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -d "$BUILD_APPS" ]] || { echo "Missing build directory: $BUILD_APPS" >&2; exit 1; }
declare -A SEEN=()
BINARIES=()
while IFS= read -r -d '' candidate; do
  name=${candidate##*/}
  [[ "$name" =~ ^starspin_[1-4]_[0-9]+([._].*)?$ ]] || continue
  [[ -f "$candidate" && -x "$candidate" ]] || continue
  binary=$(realpath -- "$candidate")
  [[ $(head -c 4 -- "$binary") == $'\x7fELF' ]] || continue
  [[ -z ${SEEN[$binary]+present} ]] || continue
  SEEN[$binary]=1
  BINARIES+=("$binary")
done < <(find "$BUILD_APPS" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
(( ${#BINARIES[@]} )) || { echo "No Starspin capture binaries found in $BUILD_APPS" >&2; exit 1; }
if [[ "$DRY_RUN" == false && $EUID -ne 0 ]]; then sudo -v; fi
for binary in "${BINARIES[@]}"; do
  if [[ "$DRY_RUN" == true ]]; then
    printf 'sudo /usr/sbin/setcap cap_net_raw=ep %q\n' "$binary"
  elif (( EUID == 0 )); then
    /usr/sbin/setcap cap_net_raw=ep "$binary"
  else
    sudo /usr/sbin/setcap cap_net_raw=ep "$binary"
  fi
done
if [[ "$DRY_RUN" == true ]]; then
  printf 'Would update %d Starspin binaries.\n' "${#BINARIES[@]}"
else
  /usr/sbin/getcap "${BINARIES[@]}"
  printf 'Updated %d Starspin binaries. Re-run after rebuilding or replacing binaries.\n' "${#BINARIES[@]}"
fi
