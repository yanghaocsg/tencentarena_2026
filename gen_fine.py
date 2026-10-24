import base64, hashlib, json

d = open('_tw_lf.py', 'rb').read()
START = 4800
CS = 150

items = []
i = START
while i < len(d):
    part = d[i:i+CS]
    b64 = base64.b64encode(part).decode()
    cum = d[:i+len(part)]
    items.append({
        'b64': b64,
        'seg_len': len(part),
        'cum_len': len(cum),
        'cum_md5': hashlib.md5(cum).hexdigest(),
    })
    i += CS

with open('_tw_fine.json', 'w') as f:
    json.dump({'start': START, 'items': items}, f, indent=0)
print('fine segments:', len(items))
print('first b64 len:', len(items[0]['b64']), 'last:', len(items[-1]['b64']))
