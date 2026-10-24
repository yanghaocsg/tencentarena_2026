import json, base64, hashlib, sys

idx = int(sys.argv[1])
d = json.load(open('_tw90.json', encoding='utf-8'))
items = d['items']

prefix = base64.b64decode(''.join(x['b64'] for x in items[:idx]))
raw = base64.b64decode(items[idx]['b64'])

half = len(raw) // 2
s0 = raw[:half]
s1 = raw[half:]

p0 = prefix + s0
p1 = p0 + s1

print("add '%s' '%s'" % (base64.b64encode(s0).decode(), hashlib.md5(p0).hexdigest()))
print("add '%s' '%s'" % (base64.b64encode(s1).decode(), hashlib.md5(p1).hexdigest()))
print('half', half, 'lens', len(p0), len(p1))
