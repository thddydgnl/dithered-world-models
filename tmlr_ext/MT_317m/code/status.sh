#!/bin/bash
# One status line for the coordinator (WORKER_RULES 3), within 5 s:
# tag=<tag> phase=<setup|smoke|main|analysis|done> done=<a>/<b> failed=<c> driver=<alive|dead|finished> eta=<UTC|-> updated=<UTC> note=<memo>
A=/workspace/atm
TAG=$(cat $A/TAG 2>/dev/null || echo "?")
PID=$(cat $A/logs/driver.pid 2>/dev/null)
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && grep -q mt14_driver "/proc/$PID/cmdline" 2>/dev/null; then DRV=alive
elif [ -f $A/out/DRIVER_FINISHED ]; then DRV=finished
else DRV=dead; fi
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ ! -f $A/SETUP_DONE ]; then
  NOTE=installing; [ -f $A/SETUP_FAILED ] && NOTE=SETUP_FAILED
  echo "tag=$TAG phase=setup done=0/0 failed=0 driver=$DRV eta=- updated=$NOW note=$NOTE"; exit 0
fi
timeout 4 python3 - "$A" "$TAG" "$DRV" "$NOW" <<'PY' 2>/dev/null || echo "tag=$TAG phase=? done=?/? failed=? driver=$DRV eta=- updated=$NOW note=status.json_unreadable"
import json, os, sys
A, tag, drv, now = sys.argv[1:5]
ph = open(f"{A}/PHASE").read().strip() if os.path.exists(f"{A}/PHASE") else ""
try:
    s = json.load(open(f"{A}/out/status.json"))
except Exception:
    s = {}
phase = ph or s.get("phase", "main")
note = s.get("note", "-") if not ph else (ph + "_running" + ("; " + s["note"] if s.get("note") else ""))
print(f"tag={tag} phase={phase} done={s.get('done', 0)}/{s.get('total', 0)} failed={s.get('failed', 0)} driver={drv} "
      f"eta={s.get('eta', '-')} updated={s.get('updated', now)} note={note}")
PY
