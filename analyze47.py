import json, base64, hashlib

d = json.load(open('_tw90.json', encoding='utf-8'))
items = d['items']

# block 47 is index 47
it = items[47]
b64 = it['b64']
print('b64 len', len(b64))
raw = base64.b64decode(b64)
print('raw len', len(raw))
print('raw repr:', repr(raw))

# verify prefix
prefix = base64.b64decode(''.join(x['b64'] for x in items[:47]))
print('prefix len', len(prefix), 'md5', hashlib.md5(prefix).hexdigest())
full = prefix + raw
print('full md5', hashlib.md5(full).hexdigest(), 'expected', it['cum_md5'])

# split this 90-byte block into two 45-byte blocks
for i, sub in enumerate([raw[:45], raw[45:]]):
    print('SUB', i, repr(sub), 'b64:', base64.b64encode(sub).decode())
