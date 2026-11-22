"""Write /workspace/atm/driver_conf.json for one td pod (tasks from plan_td.POD_TASKS). Usage: make_conf.py <tag> [workers]"""
import json
import sys

import plan_td as P

tag = sys.argv[1]
w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
A = "/workspace/atm"
conf = {"tag": tag, "tasks": P.POD_TASKS[tag.rstrip("bcdefgh")], "workers": w, "out": f"{A}/out", "big": f"{A}/big",
        "logs": f"{A}/logs", "atm": A}
json.dump(conf, open(f"{A}/driver_conf.json", "w"), indent=1)
open(f"{A}/TAG", "w").write(tag + "\n")
print(json.dumps(conf))
