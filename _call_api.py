import json, time, urllib.request

TOKEN = open(r'D:\game 2026\tencentarena_2026\_token.txt').read().strip()

def to_int32(x):
    x = int(x) & 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x

def djb(s):
    t = 5381
    for ch in s:
        shl = to_int32(to_int32(t) << 5)
        t = t + shl + ord(ch)
    return to_int32(t) & 0x7fffffff

def call(action, body, module='Competition'):
    url = f'https://tencentarena.com/api/v5/{module}/{action}'
    a = int(time.time())
    sign = djb(str(a) + TOKEN[-32:] + url.split('/')[-1])
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}',
        'x-kaiwu-auth': str(sign),
        'x-kaiwu-ts': str(a),
        'User-Agent': 'Mozilla/5.0',
        'Cookie': f'kaiwu-token={TOKEN}',
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status, resp.read().decode('utf-8', 'ignore')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'ignore')

v4file = json.load(open(r'D:\game 2026\tencentarena_2026\_v4_model_meta.json'))['file'][0]
common = {
    "domain": {"type": "competition_stage", "id": 506},
    "file_key": v4file["key"],
    "competition_team_id": 20136,
    "experiment_id": 59954,
}
srcs = [
    {"type": "ai_model"},
    {"type": "ai_model", "id": 367701},
    {"type": "competition_stage", "id": 506},
    {"source": "ai_model"},
    {"download_source": "ai_model"},
    {"type": "train"},
]
for src in srcs:
    c = dict(common); c["download_source"] = src
    st, txt = call('GetAuthDownloadURL', c)
    print(json.dumps(src), '->', st, txt[:400])
    print()
