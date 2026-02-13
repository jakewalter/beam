#!/usr/bin/env bash
# Wait for the running 3.0 forced-LSQ job to finish, then start the 3.5 job

set -eu

PIDFILE="logs/run_triangulate_force_3_0.pid"
OUTLOG="logs/run_triangulate_force_3_5.out"
PIDOUT="logs/run_triangulate_force_3_5.pid"

if [ ! -f "$PIDFILE" ]; then
  echo "PID file $PIDFILE not found; cannot watch. Exiting." >&2
  exit 1
fi

PID=$(cat "$PIDFILE")
echo "Watching PID $PID from $PIDFILE..."

while kill -0 "$PID" 2>/dev/null; do
  # process still alive
  sleep 10
done

echo "Process $PID exited; starting 3.5 forced-LSQ run..."

PYTHONPATH=. python3 scripts/run_triangulate_force_vel_full.py --config configs/bench_tri_30s_min_snr8.json --force-vel 3.5 --outdir plots/test_lsq_force_full_3_5 --time-tol 30.0 --min-snr 8.0 --lsq-vel-min 3.0 --lsq-vel-max 5.0 > "$OUTLOG" 2>&1 &
echo $! > "$PIDOUT"
echo "Started 3.5 run (PID $(cat $PIDOUT)), logging to $OUTLOG"
