"""One status line of the dw queue (WORKER_RULES 3): tag phase done failed driver eta updated note. System python."""
import json, os
ATM = '/workspace/atm'
tag = open(f'{ATM}/TAG').read().strip() if os.path.exists(f'{ATM}/TAG') else '?'


def alive(pidfile):
    try:
        os.kill(int(open(pidfile).read().strip()), 0)
        return True
    except Exception:
        return False


st = {}
try:
    st = json.load(open(f'{ATM}/out/state/status.json'))
except Exception:
    pass
fin = os.path.exists(f'{ATM}/out/state/FINISHED')
drv = 'alive' if (alive(f'{ATM}/logs/supervisor.pid') or alive(f'{ATM}/logs/driver.pid')) else ('finished' if fin else 'dead')
phase = st.get('phase', 'setup')
if fin:
    phase = 'done'
note = str(st.get('note', '') or st.get('current', '-')).replace(' ', '_')[:90]
cur = str(st.get('current', '-')).replace(' ', '_')[:60]
print(f"tag={tag} phase={phase} done={st.get('done', 0)}/{st.get('total', 0)} failed={st.get('failed', 0)} driver={drv} "
      f"eta={st.get('eta', '-')} updated={st.get('updated', '-')} note={note}|cur={cur}")
