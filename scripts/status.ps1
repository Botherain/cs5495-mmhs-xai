<#
Show what is running and how far each checkpointed job has got.

Usage: powershell -File scripts\status.ps1
#>
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$root = Split-Path -Parent $PSScriptRoot

"=== python processes ==="
Get-Process python -ErrorAction SilentlyContinue |
    Select-Object Id, StartTime, CPU, @{n = 'MB'; e = { [math]::Round($_.WorkingSet / 1MB) } } |
    Format-Table -AutoSize

"=== checkpoints ==="
Get-ChildItem (Join-Path $root 'refine-logs') -Filter '*.jsonl' -ErrorAction SilentlyContinue |
    ForEach-Object {
        $lines = (Get-Content $_.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
        "{0}: {1} records, updated {2:HH:mm:ss}" -f $_.Name, $lines, $_.LastWriteTime
    }

"=== latest log lines ==="
Get-ChildItem (Join-Path $root 'logs') -Filter '*.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 3 |
    ForEach-Object {
        "--- $($_.Name) ($($_.LastWriteTime.ToString('HH:mm:ss'))) ---"
        Get-Content $_.FullName -Tail 4 -Encoding utf8 -ErrorAction SilentlyContinue
    }
