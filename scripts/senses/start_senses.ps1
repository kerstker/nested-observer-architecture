# Her senses beyond text, started in the background (idempotent: already-running senses are left alone).
#   powershell -ExecutionPolicy Bypass -File scripts\senses\start_senses.ps1
# hearing: Whisper large-v3-turbo  -> http://127.0.0.1:8091/health
# sight:   DeepSeek-VL2-Tiny       -> http://127.0.0.1:8092/health
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

function Up([int]$port) {
    try { return (Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200 }
    catch { return $false }
}

function StartSense([string]$name, [string]$python, [string]$script, [int]$port, [string]$log) {
    if (Up $port) { Write-Host "$name: already up on :$port"; return }
    if (-not (Test-Path $python)) { Write-Host "$name: missing $python (run setup_senses.ps1 first)"; return }
    if (-not (Test-Path $script)) { Write-Host "$name: missing $script"; return }
    Start-Process -FilePath $python -ArgumentList "`"$script`" --port $port" -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput "data\$log.log" -RedirectStandardError "data\$log.err.log"
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        if (Up $port) { Write-Host "$name: up on :$port"; return }
    }
    Write-Host "$name: did not come up; see data\$log.err.log"
}

StartSense "hearing (Whisper)" ".venv-lora\Scripts\python.exe" "scripts\senses\hear_server.py" 8091 "hear_server"
StartSense "sight (DeepSeek-VL2)" ".venv-vision\Scripts\python.exe" "scripts\senses\see_server.py" 8092 "see_server"
