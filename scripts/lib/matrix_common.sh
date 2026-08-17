# Shared cell-file parsing for the matrix runners (sourced, not executed).
# Grammar: matrices/README.md — "<platform> <runtime> <model-id> <task> [k=v ...]".
#
# Usage pattern (see scripts/bench_matrix_mac.sh):
#   source "$REPO/scripts/lib/matrix_common.sh"
#   cells_for ios "$CELLS_FILE" | while read -r rt mid task rest; do
#     runs="$(cell_opt runs 4 $rest)"
#     ...
#   done
#
# cells_for strips comments/blanks, filters to one platform, drops the platform
# column, SKIPS exclude=/manual= cells (logging them to stderr so the runner can
# tee into SKIPPED.txt), and orders anchor=1 cells first (file order otherwise).

# cells_for <platform> <file...>  -> lines "<runtime> <model-id> <task> [k=v ...]"
cells_for() {
  local plat="$1"; shift
  awk -v plat="$plat" '
    { sub(/#.*/, "") }
    NF == 0 { next }
    $1 != plat { next }
    {
      excl = ""; manual = 0; anchor = 0
      for (i = 5; i <= NF; i++) {
        if ($i ~ /^exclude=/) excl = substr($i, 9)
        if ($i == "manual=1") manual = 1
        if ($i == "anchor=1") anchor = 1
      }
      line = $2; for (i = 3; i <= NF; i++) line = line " " $i
      if (excl != "") { print "CELL_SKIP " $2 " " $3 " " $4 " reason=" excl > "/dev/stderr"; next }
      if (manual)     { print "CELL_SKIP " $2 " " $3 " " $4 " reason=manual" > "/dev/stderr"; next }
      if (anchor) print "0 " line; else print "1 " line
    }
  ' "$@" | sort -s -k1,1n | cut -d' ' -f2-
}

# cell_opt <key> <default> [k=v ...]  -> value of key, or default
cell_opt() {
  local key="$1" def="$2" kv; shift 2
  for kv in "$@"; do
    case "$kv" in "$key="*) printf '%s\n' "${kv#*=}"; return ;; esac
  done
  printf '%s\n' "$def"
}
