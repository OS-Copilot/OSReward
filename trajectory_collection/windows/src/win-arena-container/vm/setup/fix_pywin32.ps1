# Repair pywin32 so C:\oem\server\main.py can import win32ui
$ErrorActionPreference = "Continue"
$py = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
$scripts = "$env:LOCALAPPDATA\Programs\Python\Python310\Scripts"
$log = "\\host.lan\Data\fix_pywin32_log.txt"

function Log($m) { $m | Tee-Object -FilePath $log -Append }

Log "=== fix start $(Get-Date) ==="
Log "Installing Visual C++ Redistributable (if missing)..."
$vc = "$env:TEMP\vc_redist.x64.exe"
try {
  Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vc -UseBasicParsing
  Start-Process -FilePath $vc -ArgumentList "/install /quiet /norestart" -Wait
  Log "VC redist installed"
} catch {
  Log "VC redist download/install failed: $_"
}

Log "Reinstalling pywin32..."
& $py -m pip uninstall -y pywin32 2>&1 | Tee-Object -FilePath $log -Append
& $py -m pip install --no-cache-dir "pywin32==306" 2>&1 | Tee-Object -FilePath $log -Append
if (Test-Path "$scripts\pywin32_postinstall.py") {
  & $py "$scripts\pywin32_postinstall.py" -install 2>&1 | Tee-Object -FilePath $log -Append
}

Log "Testing imports..."
& $py -c "import win32api, win32ui, pywinauto; print('IMPORT_OK')" 2>&1 | Tee-Object -FilePath $log -Append

Log "Restarting WinArena OnLogon task..."
Start-ScheduledTask -TaskName "WindowsArena_OnLogon" -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5
# Also start main.py directly if task fails
Start-Process -FilePath $py -ArgumentList "C:\oem\server\main.py --port 5000" -WindowStyle Minimized
Log "=== fix end $(Get-Date) ==="
