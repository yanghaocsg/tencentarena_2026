import base64, hashlib, os

base = r"D:\game 2026\tencentarena_2026\code-gorge_chase-IDE-18.1.7-v6\agent_ppo"
outdir = r"D:\game 2026\tencentarena_2026\cmds"
os.makedirs(outdir, exist_ok=True)
files = [
    ("rule_bot.py", "rb"),
    ("algorithm/algorithm.py", "al"),
    ("conf/conf.py", "cf"),
    ("workflow/train_workflow.py", "wf"),
]
CHUNK = 4000
idx = 0
md5s = {}
for path, tag in files:
    full = base + "\\" + path.replace("/", "\\")
    with open(full, "rb") as f:
        raw = f.read()
    md5s[path] = hashlib.md5(raw).hexdigest()
    b = base64.b64encode(raw).decode("ascii")
    chunks = [b[i:i+CHUNK] for i in range(0, len(b), CHUNK)]
    for j, c in enumerate(chunks):
        idx += 1
        op = ">" if j == 0 else ">>"
        cmd = "printf '%%s' '%s' %s /tmp/%s.b64" % (c, op, tag)
        with open(os.path.join(outdir, "cmd_%02d.txt" % idx), "w", encoding="ascii", newline="\n") as f:
            f.write(cmd + "\n")
    idx += 1
    with open(os.path.join(outdir, "cmd_%02d.txt" % idx), "w", encoding="ascii", newline="\n") as f:
        f.write("base64 -d /tmp/%s.b64 > %s\n" % (tag, path))
    idx += 1
    with open(os.path.join(outdir, "cmd_%02d.txt" % idx), "w", encoding="ascii", newline="\n") as f:
        f.write("md5sum %s\n" % path)

print("total cmd files:", idx)
for p, m in md5s.items():
    print(m, p)
