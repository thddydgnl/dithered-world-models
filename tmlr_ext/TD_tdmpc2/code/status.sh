#!/bin/bash
# One status line for the coordinator (WORKER_RULES 3), in well under 5 s:
# tag=<tag> phase=<setup|smoke|main|analysis|done> done=<a>/<b> failed=<c> driver=<alive|dead|finished> eta=<UTC|-> updated=<UTC> note=<memo>
A=/workspace/atm
timeout 4 python3 - "$A" <<'PY' 2>/dev/null || echo "tag=$(cat $A/TAG 2>/dev/null) phase=$(cat $A/logs/phase 2>/dev/null || echo ?) done=?/? failed=? driver=? eta=- updated=$(date -u +%Y-%m-%dT%H:%M:%SZ) note=status_script_error"
import calendar, json, os, sys, time
A = sys.argv[1]
tag = open(f"{A}/TAG").read().strip() if os.path.exists(f"{A}/TAG") else "?"
phase = open(f"{A}/logs/phase").read().strip() if os.path.exists(f"{A}/logs/phase") else ("setup" if not os.path.exists(f"{A}/SETUP_DONE") else "smoke")
if os.path.exists(f"{A}/SETUP_FAILED"):
    phase = "setup"
st = {}
try:
    st = json.load(open(f"{A}/logs/driver_state.json"))
except Exception:
    pass
alive = False
try:
    pid = int(open(f"{A}/logs/driver.pid").read().strip())
    alive = os.path.exists(f"/proc/{pid}") and b"driver.py" in open(f"/proc/{pid}/cmdline", "rb").read()
except Exception:
    pass
if alive:
    drv = "alive"
elif os.path.exists(f"{A}/logs/DRIVER_FINISHED") or st.get("driver") == "finished":
    drv = "finished"
else:
    drv = "dead" if st else "-"
note = st.get("note") or ""
if os.path.exists(f"{A}/SETUP_FAILED"):
    note = "SETUP_FAILED"
if st:
    note = (note + f" inflight={st.get('inflight')} gated={st.get('gated')} blocked={st.get('blocked')} errors={st.get('errors')} w={st.get('workers')}").strip()
age = ""
if st.get("updated"):
    try:
        age = f" state_age={int(time.time() - calendar.timegm(time.strptime(st['updated'], '%Y-%m-%dT%H:%M:%SZ')))}s"
    except Exception:
        pass
print(f"tag={tag} phase={phase} done={st.get('done', 0)}/{st.get('total', 0)} failed={st.get('failed', 0)} driver={drv} "
      f"eta={st.get('eta', '-')} updated={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} note={(note + age).replace(' ', '_')}")
PY
