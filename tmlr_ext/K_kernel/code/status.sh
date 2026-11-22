#!/bin/bash
# one status line of the E17 pod within 5 s (WORKER_RULES 3)
timeout 4 python3 /workspace/atm/code/qstat.py status 2>/dev/null || echo "tag=k.1 phase=? done=?/? failed=? driver=? eta=- updated=$(date -u +%Y-%m-%dT%H:%M:%SZ) note=qstat_timeout_or_error"
