"""Fill the E1 DINO-WM report template with the generated tables (no hand copying of numbers)."""
import re, sys
tpl, tables, v52, static, out = sys.argv[1:6]
T = open(tables).read(); V = open(v52).read(); ST = open(static).read()

def section(text, title):
    m = re.search(r'^### ' + re.escape(title) + r'\s*$', text, re.M)
    assert m, title
    rest = text[m.end():]
    n = re.search(r'^### ', rest, re.M)
    return (rest[:n.start()] if n else rest).strip()

s = open(tpl).read()
for key in re.findall(r'\{\{T:([^}]+)\}\}', s):
    s = s.replace('{{T:' + key + '}}', section(T, key))
parts = V.strip().split('\n\n')
s = s.replace('{{V:decomp}}', parts[0].strip()).replace('{{V:ref}}', parts[1].strip())
# static parts: sections 1-2 and the v5.2 definitions from the static draft
st = ST.split('\n## (v5.2)')
s = s.replace('{{STATIC}}', st[0].strip())
v52def = st[1].split('\n', 1)[1].strip()
s = s.replace('{{V52DEF}}', v52def)
assert '{{' not in s, re.findall(r'\{\{[^}]+\}\}', s)
open(out, 'w').write(s)
print('wrote', out, len(s))
