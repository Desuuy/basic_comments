<p align="center">
  <img src="09af1d89-6b93-469b-9740-241f750e067a_Git+and+GitHub.jpg" alt="Git and GitHub" width="50%">
</p>

# 🚀 MLOps & DevOps Standard Operating Procedures (SOP)

This document systematizes workflows ranging from source code management (Git) and application packaging (Docker) to model orchestration (MLflow) and cloud deployment. It serves as a comprehensive guide for AI Engineers to manage daily tasks, advanced system monitoring, and large-scale training.

---

## 1. Linux & System Fundamentals

Foundational skills for server navigation, file management, and resource monitoring.

### 1.1. Navigation & File Management
* **List files with details:** `ls -lha` includes hidden files and sizes.
* **Directory stack:** `pushd [dir]` saves the current location and moves; `popd` returns to the previous directory[cite: 6].
* **Safe directory creation:** `mkdir -p [path]` creates parent directories if they don't exist.
* **Symbolic links:** `ln -s [source] [destination]` creates a shortcut to a file or folder.
* **Permissions:** `chmod +x [script.sh]` grants execution rights to a file.

### 1.2. Resource & Hardware Monitoring
* **Interactive Monitoring:** Use `htop` for CPU/RAM and `nvtop` for detailed multi-GPU tracking.
* **GPU Status:** `nvidia-smi` provides a snapshot; `watch -n 1 nvidia-smi` runs it every second for real-time updates.
* **I/O & Network:** `iotop -oP` displays active disk read/writes; `nethogs` monitors bandwidth per process.

### 1.3. Advanced Search & Processing (Agent-style)
* Process Lookup:** `ps aux | grep [name] | grep -v grep` finds a process while excluding the search command itself.
* **Bulk Kill:** `ps aux | grep python | awk '{print $2}' | xargs kill -9` identifies and terminates all Python processes.
* **Content Search:** `grep -r "batch_size" .` recursively searches for specific strings within all files in the directory.
* **File Tracking:** `lsof -i:8080` identifies which process is using a specific network port.

---

## 2. Version Control with Git

Managing code history, collaboration, and recovery.

### 2.1. Core Workflow
* **Shallow Clone:** `git clone --depth 1 [url]` speeds up downloads by only pulling the latest commit.
* **Branching:** `git checkout -b [name]` creates and switches to a new branch.
* **Merging:** `git merge [branch_name]` integrates changes into the current branch.

### 2.2. Advanced History & Recovery
* **Visual Log:** `git log --oneline --graph --decorate` shows a compact branch history.
* **Safety Net:** `git reflog` records every action, allowing recovery of "lost" commits or resets.
* **Stashing:** `git stash` temporarily hides uncommitted changes; [cite_start]`git stash pop` restores them.
* **Clean Repo:** `git clean -fdx` removes all untracked and ignored files to reset the environment.
* **Multiple Worktrees:** `git worktree add ../[path] [branch]` allows working on different branches simultaneously in sepa.

---

## 3. Environment & Package Management

Isolating projects with a specific library version.

* **Mamba:** `mamba create -n [env] python=3.10` is a significantly faster alternative to standard Conda.
* **UV (Modern Pip):** `uv pip install [package]` offers extremely fast Python package installation.
* **Reproducibility:** `conda env export > environment.yaml` captures all packages for environment sharing.

---

## 4. Containerization with Docker

[cite_start]Packaging code and dependencies to ensure portability across environments[cite: 45, 46].

### 4.1. CLI Operations
* **GPU Access:** `docker run -it --gpus all [image]` grants the container access to all host GPUs.
* **Debugging:** `docker exec -it [id] /bin/bash` opens a live terminal inside a running container.

### 4.2. Build Optimizations (BuildKit)
* **Cache Mounts:** `RUN --mount=type=cache,target=/root/.cache/pip...` speeds up builds by caching package data.
* **Multi-stage Builds:** `FROM builder as build... FROM base COPY --from build` keeps the final production image small.

---

## 5. High-Performance Computing (Slurm)

Managing job queues for training on lar.

* **Job Submission:** `sbatch [script.slurm]` sends a job to the queue.
* **Monitoring:** `squeue -u [user]` checks the status of pending (PD) or running (R) jobs.
* **Interactive Allocation:** `salloc --gpus=1` requests live resources for development.
* **Hardware Requests:** Use `#SBATCH --gres=gpu:a100:2` in your script to request specific GPU models.

---

## 6. Cloud & AI Infrastructure

### 6.1. Cloud CLI (AWS, Azure, GCP)
* **AWS S3:** `aws s3 sync ./local s3://bucket --delete` synchronizes cloud data while removing deleted files.
* **Azure ML:** `az ml job create --file job.yaml --stream` submits training jobs and tracks logs in real-time.
  **GCP Vertex AI:** `gcloud ai custom-jobs create --local-package-path=./src` automatically packages and ships local code.

### 6.2. MLOps Tools
* **Serving:** `python -m vllm.entrypoints.api_server --model [name]` launches a high-performance LLM API.
  **Tracking:** `mlflow server` manages experiment logging and `wandb login` connects to Weights & Biases,
  *Pipelines:** `kfp dsl compile` converts Python code into Kubeflow YAML pipelines.
* **Hugging Face:** `hf upload [repo_id] [local_path]` shares models directly to the Hub.
* **Data Versioning:** `dvc add [folder]` and `dvc push` manage large datasets via cloud remotes.

---

## 7. Persistence & Efficiency

* **Tmux (Terminal Multiplexer):** `tmux new -s [name]` ensures processes keep running if your SSH connection drops.
    * `Ctrl+b, d`: Detach from the session.
    * `Ctrl+b, % / "`: Split the screen vertically or horizontally.

**💡 DevOps Tip:** Always run `git status` before committing and use `df -h` to check disk space before starting large training runs.



