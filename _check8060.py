import hashlib
data = open('_tw_lf.py', 'rb').read()
lines = open('_tw_hex22.txt').read().splitlines()
print('cur md5 8060', hashlib.md5(data[:8060]).hexdigest())
for idx in range(35, 40):
    parts = lines[idx].split("'")
    hx, want = parts[1], parts[3]
    off = 7290 + idx*22
    nxt = off+22
    real_hx = data[off:nxt].hex()
    real_md5 = hashlib.md5(data[:nxt]).hexdigest()
    print('line', idx+1, 'off', off, 'hx_ok', hx==real_hx, 'md5_ok', want==real_md5)
    print('   hx', hx)
    print('   real', real_hx)
print('content around 8060:', repr(data[8050:8090]))
