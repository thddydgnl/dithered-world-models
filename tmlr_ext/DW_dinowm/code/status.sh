#!/bin/bash
# One status line of this pod's dw queue (WORKER_RULES 3), in well under 5 s (system python, small JSON).
python3 /workspace/atm/code/status.py 2> /dev/null || echo "tag=$(cat /workspace/atm/TAG 2>/dev/null) phase=setup done=0/0 failed=0 driver=dead eta=- updated=- note=status.py error"
