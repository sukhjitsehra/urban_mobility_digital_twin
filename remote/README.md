# Running MetaST (porto + sf) on the Lambda GPU

Four scripts. Two run locally (push/pull), two run on the instance (setup/run).
The local ones are bash — on Windows run them from **Git Bash** or **WSL**, not
PowerShell.

## 0. One-time: set your connection info (local shell)
```bash
export LAMBDA_KEY="/c/path/to/lambda_key.pem"   # Git Bash path form of C:\...
export LAMBDA_HOST="ubuntu@<INSTANCE_IP>"        # Lambda default user is 'ubuntu'
```
If SSH rejects the key as "too open", in PowerShell run once:
`icacls "C:\path\to\lambda_key.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"`

## 1. Push code + data (local) — ~3.9 GB
```bash
bash remote/push_to_gpu.sh
```

## 2. Install deps (on the instance)
```bash
ssh -i "$LAMBDA_KEY" "$LAMBDA_HOST"
cd ~/mubarak/Meta_Learning
bash remote/setup_remote.sh
```

## 3. Train inside tmux (on the instance) — survives disconnects
```bash
tmux new -s metaeta 'bash remote/run_metast.sh'
# detach: Ctrl-b then d   |   reattach: tmux attach -t metaeta
# watch without attaching: tail -f ~/mubarak/Meta_Learning/run_metast.log
```

## 4. Pull results back (local)
```bash
bash remote/pull_results.sh          # -> checkpoints_remote/
```

## 5. Stop billing
Lambda bills per hour. Terminate the instance when done (coordinate with your
prof, since they launched it).

---
Notes:
- This run is `python main.py --blocks MetaST --sources porto,sf`. Per the
  decoupled workflow it trains + saves theta0 only (no transfer), so `--targets`
  is not needed. Transfer/eval is a separate `--blocks Evaluation` run later.
- Only porto + sf data dirs are transferred; each already contains its
  `ckpts_LinkETA_<city>/best.ckpt`. No `PYTHONIOENCODING` needed on Linux.
