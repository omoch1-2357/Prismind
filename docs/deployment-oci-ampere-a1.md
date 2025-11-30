# OCI Ampere A1 Deployment Guide

This guide provides comprehensive instructions for deploying and running Prismind Othello AI training on Oracle Cloud Infrastructure (OCI) Ampere A1 instances.

**Requirements Coverage:** 11.8 (OCI Ampere A1 deployment setup)

## Table of Contents

1. [Instance Specifications](#instance-specifications)
2. [Initial Setup](#initial-setup)
3. [Building Prismind](#building-prismind)
4. [Python Environment Setup](#python-environment-setup)
5. [Running Training](#running-training)
6. [Background Service Setup](#background-service-setup)
7. [Monitoring and Management](#monitoring-and-management)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

---

## Instance Specifications

### Recommended Configuration

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| Shape | VM.Standard.A1.Flex | VM.Standard.A1.Flex | ARM64 (aarch64) |
| OCPUs | 2 | 4 | Training uses rayon parallelism |
| Memory | 12 GB | 24 GB | ~600 MB for training + system |
| Storage | 50 GB | 100 GB | Checkpoints ~300 MB each |
| OS | Oracle Linux 8/9 | Ubuntu 22.04 | ARM64 build |

### OCI Free Tier

Ampere A1 instances are available in the OCI Always Free tier:
- Up to 4 OCPUs and 24 GB memory total
- Ideal for running the 1 million game training workload

### Instance Creation

1. Log into OCI Console
2. Navigate to Compute > Instances > Create Instance
3. Select:
   - **Shape:** VM.Standard.A1.Flex
   - **OCPU count:** 4
   - **Memory:** 24 GB
   - **Image:** Ubuntu 22.04 (aarch64)
4. Configure networking (allow SSH port 22)
5. Add your SSH public key
6. Create the instance

---

## Initial Setup

### 1. Connect to Instance

```bash
ssh -i ~/.ssh/your_key ubuntu@<instance-ip>
```

### 2. System Update

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    wget \
    htop \
    tmux

# Install Python 3.10+ and development headers
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev
```

### 3. Install Rust Toolchain

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Reload shell environment
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### 4. Install Maturin (Python-Rust Build Tool)

```bash
# Install maturin globally
pip3 install maturin

# Or using pipx for isolation
pip3 install pipx
pipx install maturin
```

---

## Building Prismind

### 1. Clone Repository

```bash
cd ~
git clone https://github.com/your-repo/prismind.git
cd prismind
```

### 2. ARM64-Optimized Build

The `.cargo/config.toml` is pre-configured for ARM64 optimization:

```toml
[target.aarch64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=neoverse-n1", "-C", "target-feature=+neon,+crc,+crypto"]
```

Build the release version:

```bash
# Using Makefile
make release

# Or directly with cargo (production profile with panic=abort for smaller binary)
cargo build --profile release-binary

# Or using build script
./build.sh release
```

### 3. Verify Build

```bash
# Check binary (note: release-binary profile outputs to target/release-binary/)
ls -lh target/release-binary/libprismind.so
ls -lh target/release-binary/prismind-cli

# Run tests
cargo test --release

# Run benchmarks
cargo bench
```

---

## Python Environment Setup

### 1. Create Virtual Environment

```bash
cd ~/prismind

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Prismind Python Package

```bash
# Build and install with maturin (recommended)
maturin develop --release

# Or build wheel and install
maturin build --release
pip install target/wheels/*.whl
```

### 3. Install Additional Dependencies

```bash
# Development dependencies (optional)
pip install numpy pytest matplotlib

# For monitoring dashboard
pip install rich
```

### 4. Verify Installation

```bash
python3 -c "import prismind; print(prismind.__version__)"
```

---

## Running Training

### Quick Start

```bash
# Activate virtual environment
source ~/prismind/venv/bin/activate

# Create directories
mkdir -p checkpoints logs

# Start training (1 million games)
python3 python/train.py \
    --target-games 1000000 \
    --checkpoint-interval 10000 \
    --search-time 15 \
    --epsilon 0.1 \
    --checkpoint-dir checkpoints \
    --log-dir logs
```

### Using tmux for Persistence

```bash
# Create a new tmux session
tmux new -s training

# Run training inside tmux
source ~/prismind/venv/bin/activate
python3 python/train.py --target-games 1000000

# Detach from session: Ctrl+B, then D

# Reattach later
tmux attach -t training
```

### Resume Training

```bash
python3 python/train.py \
    --resume \
    --target-games 1000000
```

---

## Background Service Setup

For production deployments, use the provided systemd service.

### 1. Install Service

```bash
# Copy service file
sudo cp deploy/prismind-training.service /etc/systemd/system/

# Edit paths if needed
sudo nano /etc/systemd/system/prismind-training.service

# Reload systemd
sudo systemctl daemon-reload
```

### 2. Configure Service

Edit `/etc/systemd/system/prismind-training.service`:

```ini
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/prismind
Environment="PATH=/home/ubuntu/prismind/venv/bin:/usr/local/bin:/usr/bin:/bin"

ExecStart=/home/ubuntu/prismind/venv/bin/python3 /home/ubuntu/prismind/python/train.py \
    --target-games 1000000 \
    --checkpoint-interval 10000 \
    --resume \
    --checkpoint-dir /home/ubuntu/prismind/checkpoints \
    --log-dir /home/ubuntu/prismind/logs
```

### 3. Enable and Start

```bash
# Enable auto-start on boot
sudo systemctl enable prismind-training

# Start the service
sudo systemctl start prismind-training

# Check status
sudo systemctl status prismind-training
```

---

## Monitoring and Management

### View Logs

```bash
# Real-time journal logs
journalctl -u prismind-training -f

# Training log files
tail -f ~/prismind/logs/training_*.log
```

### Monitor Resources

```bash
# CPU and memory usage
htop

# Disk usage
df -h

# Training progress
python3 python/monitor.py --checkpoint-dir checkpoints
```

### Service Commands

```bash
# Status
sudo systemctl status prismind-training

# Stop (graceful with checkpoint save)
sudo systemctl stop prismind-training

# Restart
sudo systemctl restart prismind-training

# View recent logs
sudo journalctl -u prismind-training --since "1 hour ago"
```

---

## Performance Optimization

### ARM64-Specific Optimizations

The build is configured with these ARM64 optimizations:

1. **Target CPU:** Neoverse-N1 (Ampere A1 processor)
2. **SIMD:** NEON vector extensions enabled
3. **CRC:** Hardware CRC32 instructions
4. **Crypto:** AES/SHA hardware acceleration

### Cargo Profile (Already Configured)

```toml
[profile.release]
opt-level = 3      # Maximum optimization
lto = true         # Link-time optimization
codegen-units = 1  # Better optimization, slower compile
panic = "abort"    # Smaller binary
```

### System Tuning

```bash
# Increase file descriptor limits (add to /etc/security/limits.conf)
* soft nofile 65536
* hard nofile 65536

# For better I/O performance (if using block storage)
sudo mount -o remount,noatime /home
```

### Expected Performance

On OCI Ampere A1 (4 OCPU, 24 GB RAM):

| Metric | Target | Notes |
|--------|--------|-------|
| Game throughput | >= 4.6 games/sec | Parallel execution |
| TD update latency | < 10ms | Per game |
| Checkpoint save | < 30 seconds | ~300 MB file |
| CPU utilization | >= 80% | Across all cores |
| Memory usage | <= 600 MB | Training state |
| Total training time | 50-60 hours | 1 million games |

---

## Troubleshooting

### Build Issues

**Error: "linking with cc failed"**
```bash
# Install development libraries
sudo apt install -y build-essential pkg-config libssl-dev
```

**Error: "maturin not found"**
```bash
# Ensure maturin is in PATH
pip3 install maturin
export PATH="$HOME/.local/bin:$PATH"
```

### Runtime Issues

**Error: "prismind module not found"**
```bash
# Rebuild and reinstall
source venv/bin/activate
maturin develop --release
```

**Error: "Out of memory"**
```bash
# Check memory usage
free -h

# Reduce transposition table size in training config
# Or increase instance memory
```

**Training hangs or crashes**
```bash
# Enable debug logging
export RUST_LOG=debug
python3 python/train.py --log-level debug ...

# Check for checkpoint corruption
python3 -c "from prismind import PyCheckpointManager; PyCheckpointManager('checkpoints').verify_latest()"
```

### Service Issues

**Service fails to start**
```bash
# Check logs
sudo journalctl -u prismind-training -n 50

# Verify paths and permissions
ls -la /home/ubuntu/prismind/
ls -la /home/ubuntu/prismind/checkpoints/
```

**Service stops unexpectedly**
```bash
# Check OOM killer
dmesg | grep -i oom

# Check service status
systemctl status prismind-training
```

---

## Backup and Recovery

### Backup Checkpoints

```bash
# Copy to object storage (OCI CLI)
oci os object put \
    --bucket-name prismind-backups \
    --file checkpoints/checkpoint_latest.bin \
    --name "checkpoint_$(date +%Y%m%d_%H%M%S).bin"

# Or use rsync to another server
rsync -avz checkpoints/ backup-server:/backups/prismind/
```

### Restore from Backup

```bash
# Download checkpoint
oci os object get \
    --bucket-name prismind-backups \
    --name checkpoint_backup.bin \
    --file checkpoints/checkpoint_latest.bin

# Resume training
python3 python/train.py --resume --target-games 1000000
```

---

## Appendix: Quick Reference

### Directory Structure

```
~/prismind/
|-- python/               # Python training scripts
|   |-- train.py         # Main training script
|   |-- monitor.py       # Monitoring dashboard
|   |-- evaluate.py      # Model evaluation
|-- checkpoints/          # Saved training states
|-- logs/                 # Training logs
|-- target/release/       # Compiled binaries
|-- venv/                 # Python virtual environment
```

### Key Commands

```bash
# Build
make release              # Or: ./build.sh release

# Install
maturin develop --release

# Train
python3 python/train.py --target-games 1000000 --resume

# Monitor
python3 python/monitor.py

# Evaluate
python3 python/evaluate.py --checkpoint checkpoints/latest.bin

# Service
sudo systemctl {start|stop|status|restart} prismind-training
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Logging level (trace/debug/info/warn/error) | info |
| `RUST_BACKTRACE` | Enable backtraces (1 or full) | 0 |
| `RAYON_NUM_THREADS` | Number of worker threads | auto |

---

*Last updated: November 2025*
