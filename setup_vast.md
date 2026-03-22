# Vast.ai Setup (General, No Secrets)

Video: [VastAI Setup Youtube](https://www.youtube.com/watch?v=tSc9Jdt9bXs&embeds_referring_euri=https%3A%2F%2Fclassroom.google.com%2F&source_ve_path=Mjg2NjY)

--- 

## Notes (security)
- Do not paste API keys, passwords, or private SSH keys into this file.
- Treat any `VAST_API_KEY`, SSH private key, or Docker credentials as secrets.
- This guide is written as a template; replace placeholders like `YOUR_API_KEY` and `YOUR_IMAGE`.

---

## 1) Install Vast.ai CLI

### Windows (PowerShell)
```powershell
py -m pip install vastai
# hoac
python -m pip install vastai
```

### Linux / macOS
```bash
pip install vastai
```

---

## 2) Configure API Key (CLI auth)

Set your API key (get it from the Vast.ai console):
```bash
vastai set api-key YOUR_API_KEY
```

If you ever need to inspect what you have configured:
```bash
vastai show api-keys
```

---

## 3) Create and (optionally) attach an SSH key

Generate a new SSH key pair:
```bash
vastai create ssh-key
```

List keys to confirm what exists:
```bash
vastai show ssh-keys
```

Attach an SSH key to a specific instance (fill in your instance id and key name):
```bash
vastai attach ssh INSTANCE_ID SSH_KEY_NAME
```

---

## 4) Search offers (pick a GPU)

Use `vastai search offers` with a quoted query:
```bash
vastai search offers 'reliability > 0.98 num_gpus=1 gpu_name=RTX_3090 rented=False'
```

More examples:
```bash
vastai search offers 'compute_cap > 610 total_flops > 5 datacenter=True'
vastai search offers 'reliability > 0.99 num_gpus>=4 verified=False rented=any' -o 'num_gpus-'
```

Tip: If you use `>` or `<` in the CLI, wrap the query in quotes.

---

## 5) Create an instance from an offer id

Create a new instance from an offer id returned by `search offers`.
You usually must pass `--image` (and optionally `--ssh` / `--direct`):
```bash
vastai create instance OFFER_ID --image YOUR_IMAGE --disk 40 --ssh --direct
```

If you want to provide environment variables and ports, use `--env`:
```bash
vastai create instance OFFER_ID --image YOUR_IMAGE --disk 40 --ssh --direct --env '-e TZ=UTC -p 22:22'
```

After creation, you will get a new instance id back in the output.

---

## 6) Connect via SSH / copy files

Get an SSH connection string for an instance:
```bash
vastai ssh-url INSTANCE_ID
```

Then run the resulting `ssh ...` command in your terminal.

For copying files:
```bash
vastai scp-url INSTANCE_ID
```

---

## 7) Quick start template (inside the instance)

Once connected, a common workflow is:
```bash
cd ~
git clone https://github.com/YOUR_ORG/YOUR_REPO.git
cd YOUR_REPO

# create venv / install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run your job
python train.py
```

---

## 8) Stop / start / destroy (optional)

Stop a running instance (data is preserved):
```bash
vastai stop instance INSTANCE_ID
```

Start again later:
```bash
vastai start instance INSTANCE_ID
```

Destroy (irreversible; deletes data):
```bash
vastai destroy instance INSTANCE_ID
```

