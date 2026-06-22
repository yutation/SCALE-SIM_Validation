#!/bin/sh
set -eu

APPLY=0
JOURNAL_SIZE="${JOURNAL_SIZE:-512M}"
USER_HOME="${USER_HOME:-/home/Owner}"

usage() {
  cat <<'EOF'
Clean common VM garbage: oversized system logs, old /tmp logs, and user caches.

Usage:
  script/clean_vm_space.sh          # dry run
  script/clean_vm_space.sh --apply  # actually delete/truncate files

Environment:
  JOURNAL_SIZE=512M   systemd journal retention target
  USER_HOME=/home/Owner
EOF
}

for arg in "$@"; do
  case "$arg" in
    --apply)
      APPLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  if [ "$APPLY" -eq 1 ]; then
    echo "+ $*"
    "$@"
  else
    echo "[dry-run] $*"
  fi
}

section() {
  printf '\n== %s ==\n' "$1"
}

section "Disk usage before"
df -h /

section "Cleanup plan"
echo "- Limit systemd journal to ${JOURNAL_SIZE}"
echo "- Remove rotated /var/log/syslog and /var/log/kern.log files"
echo "- Truncate current /var/log/syslog and /var/log/kern.log"
echo "- Remove old TPU/main logs and temporary setup/cache files from /tmp"
echo "- Remove pip and maple-jax compile caches from ${USER_HOME}/.cache"

section "System logs"
run sudo -n journalctl --vacuum-size="${JOURNAL_SIZE}"
run sudo -n rm -f \
  /var/log/syslog.[0-9]* \
  /var/log/kern.log.[0-9]* \
  /var/log/syslog.*.gz \
  /var/log/kern.log.*.gz
run sudo -n truncate -s 0 /var/log/syslog /var/log/kern.log

section "Temporary logs and caches"
run sudo -n rm -f \
  /tmp/main.t1v-*.log.* \
  /tmp/*_setup.pkl.bak \
  /tmp/test_setup_cache.pkl
run sudo -n rm -rf \
  /tmp/tpu_logs \
  /tmp/pip-* \
  /tmp/pymp-* \
  /tmp/pytest-of-Owner

section "User caches"
run rm -rf \
  "${USER_HOME}/.cache/pip" \
  "${USER_HOME}/.cache/maple-jax-compile"

section "Disk usage after"
df -h /

if [ "$APPLY" -eq 0 ]; then
  cat <<'EOF'

Dry run only. Re-run with --apply to clean:
  script/clean_vm_space.sh --apply
EOF
fi
