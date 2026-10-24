import json, base64, hashlib

d = json.load(open('_tw90.json', encoding='utf-8'))
items = d['items']

# full block 66
full_prefix = base64.b64decode(''.join(x['b64'] for x in items[:66]))
raw = base64.b64decode(items[66]['b64'])

# first 45 bytes already written (current length 5985)
written = full_prefix + raw[:45]
assert len(written) == 5985

rest = raw[45:]
print('rest len', len(rest))
sizes = [22, 23]
cur = written
off = 45
for sz in sizes:
    seg = rest[off-45:off-45+sz] if False else None
# simpler:
seg1 = rest[:22]
seg2 = rest[22:]
for seg in [seg1, seg2]:
    cur = cur + seg
    print("add '%s' '%s'" % (base64.b64encode(seg).decode(), hashlib.md5(cur).hexdigest()))
print('final len', len(cur), hashlib.md5(cur).hexdigest())
