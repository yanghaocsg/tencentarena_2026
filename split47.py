import json, base64, hashlib

d = json.load(open('_tw90.json', encoding='utf-8'))
items = d['items']

prefix = base64.b64decode(''.join(x['b64'] for x in items[:47]))
raw = base64.b64decode(items[47]['b64'])

s0 = raw[:45]
s1 = raw[45:]

p0 = prefix + s0
p1 = p0 + s1

print("add '%s' '%s'" % (base64.b64encode(s0).decode(), hashlib.md5(p0).hexdigest()))
print("add '%s' '%s'" % (base64.b64encode(s1).decode(), hashlib.md5(p1).hexdigest()))
print('len p0', len(p0), 'len p1', len(p1))
