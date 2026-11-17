<#
Launch a long-running python job detached from the opencode session.

Background shells die when opencode exits; this wrapper starts a real process with
its own log files so an interrupted session costs at most one checkpoint line.

Usage:
  powershell -File scripts\run_detached.ps1 -Target r005_text_aopc.py [-Args "--limit 50"]
#>
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [string]$Args = ""
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONFAULTHANDLER = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
if (-not $env:MMHS_MODEL_DIR) {
    $env:MMHS_MODEL_DIR = 'C:\Users\123\AppData\Local\Temp\opencode\mmhs_model'
}

$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$name = [System.IO.Path]::GetFileNameWithoutExtension($Target)
$stdout = Join-Path $logs "$name.out.log"
$stderr = Join-Path $logs "$name.err.log"
$script = Join-Path $root "scripts\$Target"

$argumentList = "`"$script`""
if ($Args -ne "") { $argumentList = "$argumentList $Args" }

$process = Start-Process -FilePath 'D:\HKStudy\.venv\Scripts\python.exe' `
    -ArgumentList $argumentList -WorkingDirectory $root `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -WindowStyle Hidden -PassThru

"started $Target pid=$($process.Id)"
"stdout: $stdout"
"stderr: $stderr"
