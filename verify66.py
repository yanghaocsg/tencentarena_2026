import json, base64, hashlib

d = json.load(open('_tw90.json', encoding='utf-8'))
items = d['items']

print('idx66 cum_len in json:', items[66]['cum_len'])
print('idx66 cum_md5 in json:', items[66]['cum_md5'])

prefix = base64.b64decode(''.join(x['b64'] for x in items[:66]))
print('prefix len', len(prefix), 'md5', hashlib.md5(prefix).hexdigest())

raw = base64.b64decode(items[66]['b64'])
print('raw len', len(raw), 'repr:', repr(raw))

full = prefix + raw
print('full len', len(full), 'md5', hashlib.md5(full).hexdigest())
