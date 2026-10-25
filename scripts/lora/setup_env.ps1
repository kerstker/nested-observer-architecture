# Sets up the environment for Noa's own model (separate from the app's Python), then downloads the base.
#   powershell -ExecutionPolicy Bypass -File scripts\lora\setup_env.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }
$py = Join-Path $root ".venv-lora\Scripts\python.exe"
if (-not (Test-Path $py)) { uv venv .venv-lora --python 3.12 }

Write-Host "== PyTorch (CUDA 13.0 build, for RTX 50 / driver 610) =="
# Pin the CUDA build: an unpinned install from a CUDA index can silently fall back to a CPU-only wheel
uv pip install --python $py "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130 --reinstall-package torch; Check "torch"
Write-Host "== transformers / peft / modelscope =="
uv pip install --python $py "transformers>=4.51" peft accelerate datasets safetensors modelscope; Check "libraries"
& $py -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, 'peft', peft.__version__)"; Check "CUDA check"

Write-Host "== DeepSeek-R1-0528-Qwen3-8B (ModelScope, ~16 GB) =="
& $py -c "from modelscope import snapshot_download; p = snapshot_download('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B', local_dir='models/DeepSeek-R1-0528-Qwen3-8B'); print('saved to', p)"
Write-Host "== done =="
