<p align="center">
  <img src="09af1d89-6b93-469b-9740-241f750e067a_Git+and+GitHub.jpg" alt="Git and GitHub" width="50%">
</p>

# 🚀 MLOps & DevOps Standard Operating Procedures (SOP)

[cite_start]This document systematizes workflows ranging from source code management (Git) and application packaging (Docker) to model orchestration (MLflow) and cloud deployment[cite: 3, 59, 60]. [cite_start]It serves as a comprehensive guide for AI Engineers to manage daily tasks, advanced system monitoring, and large-scale training[cite: 3, 5, 8].

---

## 1. Linux & System Fundamentals

[cite_start]Foundational skills for server navigation, file management, and resource monitoring[cite: 4, 5].

### 1.1. Navigation & File Management
* [cite_start]**List files with details:** `ls -lha` includes hidden files and sizes[cite: 6].
* **Directory stack:** `pushd [dir]` saves the current location and moves; [cite_start]`popd` returns to the previous directory[cite: 6].
* [cite_start]**Safe directory creation:** `mkdir -p [path]` creates parent directories if they don't exist[cite: 6].
* [cite_start]**Symbolic links:** `ln -s [source] [destination]` creates a shortcut to a file or folder[cite: 6].
* [cite_start]**Permissions:** `chmod +x [script.sh]` grants execution rights to a file[cite: 6].

### 1.2. Resource & Hardware Monitoring
* [cite_start]**Interactive Monitoring:** Use `htop` for CPU/RAM and `nvtop` for detailed multi-GPU tracking[cite: 7].
* **GPU Status:** `nvidia-smi` provides a snapshot; [cite_start]`watch -n 1 nvidia-smi` runs it every second for real-time updates[cite: 7].
* **I/O & Network:** `iotop -oP` displays active disk read/writes; [cite_start]`nethogs` monitors bandwidth per process[cite: 7].

### 1.3. Advanced Search & Processing (Agent-style)
* [cite_start]**Process Lookup:** `ps aux | grep [name] | grep -v grep` finds a process while excluding the search command itself[cite: 11].
* [cite_start]**Bulk Kill:** `ps aux | grep python | awk '{print $2}' | xargs kill -9` identifies and terminates all Python processes[cite: 11].
* [cite_start]**Content Search:** `grep -r "batch_size" .` recursively searches for specific strings within all files in the directory[cite: 13].
* [cite_start]**File Tracking:** `lsof -i:8080` identifies which process is using a specific network port[cite: 15].

---

## 2. Version Control with Git

[cite_start]Managing code history, collaboration, and recovery[cite: 18, 19].

### 2.1. Core Workflow
* [cite_start]**Shallow Clone:** `git clone --depth 1 [url]` speeds up downloads by only pulling the latest commit[cite: 22].
* [cite_start]**Branching:** `git checkout -b [name]` creates and switches to a new branch[cite: 22].
* [cite_start]**Merging:** `git merge [branch_name]` integrates changes into the current branch[cite: 22].

### 2.2. Advanced History & Recovery
* [cite_start]**Visual Log:** `git log --oneline --graph --decorate` shows a compact branch history[cite: 24].
* [cite_start]**Safety Net:** `git reflog` records every action, allowing recovery of "lost" commits or resets[cite: 25].
* **Stashing:** `git stash` temporarily hides uncommitted changes; [cite_start]`git stash pop` restores them[cite: 25].
* [cite_start]**Clean Repo:** `git clean -fdx` removes all untracked and ignored files to reset the environment[cite: 25].
* [cite_start]**Multiple Worktrees:** `git worktree add ../[path] [branch]` allows working on different branches simultaneously in separate folders[cite: 25].

---

## 3. Environment & Package Management

[cite_start]Isolating projects with specific library versions[cite: 26, 27].

* [cite_start]**Mamba:** `mamba create -n [env] python=3.10` is a significantly faster alternative to standard Conda[cite: 31].
* [cite_start]**UV (Modern Pip):** `uv pip install [package]` offers extremely fast Python package installation[cite: 33].
* [cite_start]**Reproducibility:** `conda env export > environment.yaml` captures all packages for environment sharing[cite: 31].

---

## 4. Containerization with Docker

[cite_start]Packaging code and dependencies to ensure portability across environments[cite: 45, 46].

### 4.1. CLI Operations
* [cite_start]**GPU Access:** `docker run -it --gpus all [image]` grants the container access to all host GPUs[cite: 48].
* [cite_start]**Debugging:** `docker exec -it [id] /bin/bash` opens a live terminal inside a running container[cite: 48].

### 4.2. Build Optimizations (BuildKit)
* [cite_start]**Cache Mounts:** `RUN --mount=type=cache,target=/root/.cache/pip...` speeds up builds by caching package data[cite: 50].
* [cite_start]**Multi-stage Builds:** `FROM builder as build... FROM base COPY --from build` keeps the final production image small[cite: 50].

---

## 5. High-Performance Computing (Slurm)

[cite_start]Managing job queues for training on large GPU clusters[cite: 38, 39].

* [cite_start]**Job Submission:** `sbatch [script.slurm]` sends a job to the queue[cite: 41].
* [cite_start]**Monitoring:** `squeue -u [user]` checks the status of pending (PD) or running (R) jobs[cite: 42].
* [cite_start]**Interactive Allocation:** `salloc --gpus=1` requests live resources for development[cite: 42].
* [cite_start]**Hardware Requests:** Use `#SBATCH --gres=gpu:a100:2` in your script to request specific GPU models[cite: 44].

---

## 6. Cloud & AI Infrastructure

### 6.1. Cloud CLI (AWS, Azure, GCP)
* [cite_start]**AWS S3:** `aws s3 sync ./local s3://bucket --delete` synchronizes cloud data while removing deleted files[cite: 54].
* [cite_start]**Azure ML:** `az ml job create --file job.yaml --stream` submits training jobs and tracks logs in real-time[cite: 56].
* [cite_start]**GCP Vertex AI:** `gcloud ai custom-jobs create --local-package-path=./src` automatically packages and ships local code[cite: 58].

### 6.2. MLOps Tools
* [cite_start]**Serving:** `python -m vllm.entrypoints.api_server --model [name]` launches a high-performance LLM API[cite: 62].
* [cite_start]**Tracking:** `mlflow server` manages experiment logging and `wandb login` connects to Weights & Biases[cite: 62, 64].
* [cite_start]**Pipelines:** `kfp dsl compile` converts Python code into Kubeflow YAML pipelines[cite: 66].
* [cite_start]**Hugging Face:** `hf upload [repo_id] [local_path]` shares models directly to the Hub[cite: 69].
* [cite_start]**Data Versioning:** `dvc add [folder]` and `dvc push` manage large datasets via cloud remotes[cite: 72, 73].

---

## 7. Persistence & Efficiency

* [cite_start]**Tmux (Terminal Multiplexer):** `tmux new -s [name]` ensures processes keep running if your SSH connection drops[cite: 34, 35].
    * [cite_start]`Ctrl+b, d`: Detach from the session[cite: 37].
    * [cite_start]`Ctrl+b, % / "`: Split the screen vertically or horizontally[cite: 37].

[cite_start]**💡 DevOps Tip:** Always run `git status` before committing [cite: 22] [cite_start]and use `df -h` to check disk space before starting large training runs[cite: 17].
