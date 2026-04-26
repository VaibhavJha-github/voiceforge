# VoiceForge Setup Guide — Windows PC (RTX 4070 Ti)

## Prerequisites
- Windows 10/11 with WSL2 installed
- NVIDIA drivers updated (Game Ready or Studio)
- RTX 4070 Ti (12GB VRAM)

## Step 1: Set Up WSL2 with CUDA

Open PowerShell as Admin:
```powershell
wsl --install -d Ubuntu-24.04
```

Restart if needed, then open Ubuntu terminal.

## Step 2: Install NVIDIA CUDA in WSL2

```bash
# CUDA toolkit (WSL2 uses the Windows NVIDIA driver, just need toolkit)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Verify
nvidia-smi
```

## Step 3: Install Python + Dependencies

```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg portaudio19-dev libsox-dev

# Create venv
python3.12 -m venv ~/voiceforge-env
source ~/voiceforge-env/bin/activate
```

## Step 4: Clone & Install Fish-Speech

```bash
cd ~
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Install with CUDA support
pip install -e ".[cu126]"
```

## Step 5: Download the Model

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Download S2-Pro model (~8GB)
huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## Step 6: Run VoiceForge

```bash
cd ~/fish-speech
source ~/voiceforge-env/bin/activate

# With --half flag for 12GB VRAM
python tools/voiceforge_app.py --half
```

Open browser: **http://localhost:7860**

## Step 7: Clone Your Voices

1. Go to **Voice Setup** tab
2. Enter name: `PETER`
3. Upload 10-30 seconds of Peter Griffin voice audio
4. Type exactly what's being said in that audio clip
5. Click **Clone Voice**
6. Repeat for `STEWIE`

## Step 8: Generate Audio

1. Go to **Generate** tab
2. Paste your script (or upload .txt file):
   ```
   === VIDEO 1 ===
   PETER: Hey Stewie, did you know...
   STEWIE: Wait, what?
   ...
   ```
3. Click **Parse & Preview** to check everything looks right
4. Click **Generate All Audio**
5. Download the ZIP with all audio files

## Troubleshooting

**Out of memory (OOM):**
- Make sure you're using `--half` flag
- Close other GPU-heavy apps (games, etc.)
- Try reducing `max_new_tokens` in the UI advanced settings

**Slow generation:**
- Add `--compile` flag for ~2x speedup (first run is slow, subsequent runs fast)
- `python tools/voiceforge_app.py --half --compile`

**WSL2 can't see GPU:**
- Update Windows NVIDIA drivers
- Run `nvidia-smi` in WSL2 — if it fails, drivers need updating
- Make sure WSL2 is version 2: `wsl -l -v`
