import base64, hashlib, json

d = open('_tw_lf.py', 'rb').read()
CS = 90
items = []
i = 0
while i < len(d):
    part = d[i:i+CS]
    b64 = base64.b64encode(part).decode()
    cum = d[:i+len(part)]
    items.append({'b64': b64, 'seg_len': len(part), 'cum_len': len(cum),
                  'cum_md5': hashlib.md5(cum).hexdigest()})
    i += CS
json.dump({'items': items}, open('_tw90.json', 'w'))
print('segments:', len(items))
