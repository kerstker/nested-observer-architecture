# Her hearing beyond text: Whisper large-v3-turbo.
#   powershell -ExecutionPolicy Bypass -File scripts\senses\setup_senses.ps1
# Sight needs no local model any more: the model looks at pictures in the conversation itself
# (qq.media() -> DeepSeekEngine.with_media, see docs/图片理解.md). Only hearing is set up here.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
# PyPI itself is very slow from here; the Alibaba Cloud mirror serves the same packages
$pypi = "https://mirrors.aliyun.com/pypi/simple/"
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

Write-Host "== hearing: Whisper large-v3-turbo (ModelScope, ~1.6 GB) =="
$lora = Join-Path $root ".venv-lora\Scripts\python.exe"
& $lora -c "from modelscope import snapshot_download; print(snapshot_download('openai-mirror/whisper-large-v3-turbo', local_dir='models/whisper-large-v3-turbo'))"; Check "whisper download"
uv pip install --python $lora soundfile --index-url $pypi; Check "soundfile"
Write-Host "== done =="
