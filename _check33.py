import hashlib
data = open('_tw_lf.py', 'rb').read()
print('len', len(data))
chunk = data[7994:8016]
print('chunk len', len(chunk))
print('chunk hex', chunk.hex())
print('md5 8016', hashlib.md5(data[:8016]).hexdigest())
print('md5 7994', hashlib.md5(data[:7994]).hexdigest())
# 显示该字节附近内容
print(repr(data[7985:8030]))
