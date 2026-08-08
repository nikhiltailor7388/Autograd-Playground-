$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $root 'frontend'
$envFile = Join-Path $frontend '.env'
$backendCandidates = @(
    (Join-Path $root 'build4\backend\autograd_server.exe'),
    (Join-Path $root 'build\backend\autograd_server.exe'),
    (Join-Path $root 'build-new\backend\autograd_server.exe')
)
$backendExe = $backendCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*?)\s*$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}

Write-Host 'Starting Autograd full stack...' -ForegroundColor Cyan

if ($backendExe) {
    $backendProcess = Start-Process -FilePath $backendExe -ArgumentList '8080' -WorkingDirectory $root -PassThru
    Write-Host "C++ backend started: $backendExe" -ForegroundColor Green
} else {
    Write-Warning 'C++ backend executable was not found. Starting the frontend mock API instead.'
    Write-Warning 'Install a modern C++ compiler, rebuild with CMake, and rerun this launcher for the real backend.'
}

if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
    Write-Host 'Installing frontend dependencies...' -ForegroundColor Yellow
    Push-Location $frontend
    try { & npm.cmd install } finally { Pop-Location }
}

$frontendProcess = Start-Process -FilePath 'cmd.exe' -ArgumentList "/c set PORT=3000&& node `"$(Join-Path $frontend 'server.js')`"" -WorkingDirectory $frontend -PassThru
Start-Sleep -Milliseconds 700
Start-Process 'http://localhost:3000'

Write-Host ''
Write-Host 'Frontend: http://localhost:3000' -ForegroundColor Green
Write-Host 'Backend:  http://localhost:8080/api/health' -ForegroundColor Green
Write-Host 'Close this window to stop the servers.' -ForegroundColor DarkGray

try {
    while (-not $frontendProcess.HasExited) {
        Start-Sleep -Seconds 1
    }
} finally {
    if ($backendProcess -and -not $backendProcess.HasExited) { Stop-Process $backendProcess.Id -Force }
    if ($frontendProcess -and -not $frontendProcess.HasExited) { Stop-Process $frontendProcess.Id -Force }
}