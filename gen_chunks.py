import base64, hashlib, json

d = open('_tw_lf.py', 'rb').read()
print('total bytes:', len(d))

CHUNK = 600
chunks = []
cum_md5 = []
cum_len = []
for i in range(0, len(d), CHUNK):
    part = d[i:i+CHUNK]
    chunks.append(base64.b64encode(part).decode())
    cum = d[:i+len(part)]
    cum_md5.append(hashlib.md5(cum).hexdigest())
    cum_len.append(len(cum))

out = {'chunk_size': CHUNK, 'total': len(d), 'n': len(chunks),
       'chunks': chunks, 'cum_md5': cum_md5, 'cum_len': cum_len}
with open('_tw_chunks.json', 'w') as f:
    json.dump(out, f, indent=0)
print('chunks:', len(chunks))
print('b64 len each:', len(chunks[0]), 'last:', len(chunks[-1]))
