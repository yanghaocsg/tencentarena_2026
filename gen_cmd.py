import json

d = json.load(open('_tw90.json', encoding='utf-8'))
items = d['items']

with open('_tw_cmds.txt', 'w', encoding='utf-8', newline='') as f:
    for it in items:
        f.write("add '%s' '%s'\n" % (it['b64'], it['cum_md5']))

print('total', len(items))
print('first b64 len', len(items[0]['b64']))
print('last b64 len', len(items[-1]['b64']))
