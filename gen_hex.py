import hashlib

data = open('_tw_lf.py', 'rb').read()
print('total', len(data))

START = 6030
CHUNK = 45

lines = []
off = START
while off < len(data):
    seg = data[off:off+CHUNK]
    h = seg.hex()
    cum = data[:off+len(seg)]
    md5 = hashlib.md5(cum).hexdigest()
    lines.append("addh '%s' '%s'" % (h, md5))
    off += len(seg)

with open('_tw_hex.txt', 'w', encoding='utf-8', newline='') as f:
    f.write('\n'.join(lines) + '\n')

print('chunks', len(lines))
print('first hex len', len(data[START:START+CHUNK])*2)
