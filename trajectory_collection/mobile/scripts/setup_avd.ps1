# Sets up the AndroidWorldAvd emulator image on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup_avd.ps1
#
# What it does:
#   1. Locates the Android SDK.
#   2. Locates a JDK (Android Studio's bundled "jbr" if JAVA_HOME is unset).
#   3. Installs required SDK components (emulator, platform-tools,
#      android-33 Google APIs system image) via sdkmanager.
#   4. Downloads AndroidWorldAvd.avd.zip from Hugging Face (resumable).
#   5. Unzips it into %USERPROFILE%\.android\avd.
#   6. Writes AndroidWorldAvd.ini and fills the {{ANDROID_SDK_ROOT}}
#      placeholder in config.ini with your local SDK path.
#   7. Verifies the AVD is visible to the emulator.
#
# Re-running is safe: finished steps are skipped. Use -Force to redo the
# download and unzip.

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$AvdName = "AndroidWorldAvd"
$ZipUrl = "https://huggingface.co/datasets/Lemaqwq/AndroidWorldAvd/resolve/main/AndroidWorldAvd.avd.zip"
$SystemImage = "system-images;android-33;google_apis;x86_64"

function Fail($Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Step($Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# --- 1. Locate the Android SDK -------------------------------------------
Step "Locating Android SDK"
$SdkRoot = $env:ANDROID_SDK_ROOT
if (-not $SdkRoot) { $SdkRoot = $env:ANDROID_HOME }
if (-not $SdkRoot) { $SdkRoot = Join-Path $env:LOCALAPPDATA "Android\Sdk" }
if (-not (Test-Path $SdkRoot)) {
    Fail ("Android SDK not found at '$SdkRoot'. Install Android Studio first " +
          "(https://developer.android.com/studio), or set ANDROID_SDK_ROOT.")
}
Write-Host "    SDK: $SdkRoot"

# --- 2. Locate a JDK ------------------------------------------------------
# sdkmanager.bat is a Java program: without JAVA_HOME (or java on PATH) it
# aborts before doing anything. Android Studio bundles a JDK as "jbr", but
# neither the installer nor the SDK puts it on PATH.
Step "Locating a JDK for sdkmanager"
function Test-Jdk($Dir) {
    return $Dir -and (Test-Path (Join-Path $Dir "bin\java.exe"))
}
$JavaHome = $null
if (Test-Jdk $env:JAVA_HOME) {
    $JavaHome = $env:JAVA_HOME
} elseif (Get-Command java -ErrorAction SilentlyContinue) {
    $JavaHome = "PATH"
} else {
    # Android Studio can be installed on any drive, so read its location from
    # the registry rather than guessing Program Files.
    $StudioDirs = @()
    foreach ($Key in @("HKLM:\SOFTWARE\Android Studio", "HKCU:\SOFTWARE\Android Studio")) {
        try {
            $Path = (Get-ItemProperty $Key -ErrorAction Stop).Path
            if ($Path) { $StudioDirs += $Path }
        } catch {}
    }
    $StudioDirs += @(
        (Join-Path $env:ProgramFiles "Android\Android Studio"),
        (Join-Path ${env:ProgramFiles(x86)} "Android\Android Studio"),
        (Join-Path $env:LOCALAPPDATA "Programs\Android Studio")
    )
    foreach ($Dir in $StudioDirs) {
        foreach ($Sub in @("jbr", "jre")) {
            $Candidate = Join-Path $Dir $Sub
            if (Test-Jdk $Candidate) { $JavaHome = $Candidate; break }
        }
        if ($JavaHome) { break }
    }
}
if (-not $JavaHome) {
    Fail ("No JDK found. sdkmanager needs one. Install Android Studio " +
          "(it bundles a JDK), or set JAVA_HOME to a JDK 17+ install.")
}
if ($JavaHome -eq "PATH") {
    Write-Host "    JDK: using 'java' from PATH"
} else {
    Write-Host "    JDK: $JavaHome"
    $env:JAVA_HOME = $JavaHome
    $env:PATH = (Join-Path $JavaHome "bin") + ";" + $env:PATH
}

# --- 3. Install SDK components -------------------------------------------
Step "Installing SDK components (emulator, platform-tools, system image)"
$SdkManager = Get-ChildItem -Path (Join-Path $SdkRoot "cmdline-tools") `
    -Filter "sdkmanager.bat" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $SdkManager) {
    Fail ("sdkmanager.bat not found under $SdkRoot\cmdline-tools. In Android " +
          "Studio: Settings > Languages & Frameworks > Android SDK > SDK Tools " +
          "> check 'Android SDK Command-line Tools' > Apply. Then re-run.")
}
# Accept licenses non-interactively, then install what the AVD needs.
@("y") * 20 | & $SdkManager --licenses | Out-Null
& $SdkManager "emulator" "platform-tools" $SystemImage
if ($LASTEXITCODE -ne 0) { Fail "sdkmanager failed (exit $LASTEXITCODE)." }

# --- 4. Download the snapshot zip ----------------------------------------
$AvdHome = Join-Path $env:USERPROFILE ".android\avd"
New-Item -ItemType Directory -Path $AvdHome -Force | Out-Null
$ZipPath = Join-Path $env:TEMP "AndroidWorldAvd.avd.zip"
$AvdDir = Join-Path $AvdHome "$AvdName.avd"

if ((Test-Path $AvdDir) -and -not $Force) {
    Step "AVD folder already exists, skipping download and unzip ($AvdDir)"
} else {
    Step "Downloading snapshot (several GB; resumable, re-run if interrupted)"
    # curl.exe ships with Windows 10+; -C - resumes partial downloads.
    & curl.exe -L -C - --fail --retry 3 -o $ZipPath $ZipUrl
    if ($LASTEXITCODE -ne 0) {
        Fail "Download failed. Re-run the script to resume, or fetch manually: $ZipUrl"
    }

    # --- 5. Unzip into the AVD home --------------------------------------
    Step "Unzipping into $AvdHome"
    if (Test-Path $AvdDir) { Remove-Item -Recurse -Force $AvdDir }
    Expand-Archive -Path $ZipPath -DestinationPath $AvdHome -Force
    # Tolerate a zip whose top-level folder carries a suffix, e.g. _2.
    if (-not (Test-Path $AvdDir)) {
        $Candidate = Get-ChildItem $AvdHome -Directory -Filter "$AvdName*.avd" |
            Select-Object -First 1
        if ($Candidate) { Rename-Item $Candidate.FullName $AvdDir }
    }
    if (-not (Test-Path (Join-Path $AvdDir "config.ini"))) {
        Fail "Unzip did not produce $AvdDir\config.ini - unexpected zip layout."
    }
    Remove-Item $ZipPath -ErrorAction SilentlyContinue
}

# --- 6. Write the .ini and fill in the SDK placeholder -------------------
Step "Writing $AvdName.ini and patching config.ini"
@"
avd.ini.encoding=UTF-8
path=$AvdDir
path.rel=avd\$AvdName.avd
target=android-33
"@ | Set-Content -Path (Join-Path $AvdHome "$AvdName.ini") -Encoding UTF8

$ConfigPath = Join-Path $AvdDir "config.ini"
$Config = Get-Content $ConfigPath -Raw
$SkinDir = Join-Path $SdkRoot "skins\pixel_6a"
if (Test-Path $SkinDir) {
    $Config = $Config.Replace("{{ANDROID_SDK_ROOT}}/skins/pixel_6a", $SkinDir)
} else {
    # No pixel_6a skin in this SDK: fall back to a plain resolution skin so
    # the emulator boots instead of failing on a missing skin folder.
    Write-Host "    pixel_6a skin not found, using 1080x2400 fallback"
    $Config = $Config -replace "skin\.path=.*", "skin.path=1080x2400"
    $Config = $Config -replace "skin\.name=.*", "skin.name=1080x2400"
}
Set-Content -Path $ConfigPath -Value $Config -Encoding UTF8

# --- 7. Verify -----------------------------------------------------------
Step "Verifying"
$Emulator = Join-Path $SdkRoot "emulator\emulator.exe"
$Avds = & $Emulator -list-avds 2>$null
if ($Avds -notcontains $AvdName) {
    Fail "'$AvdName' not listed by 'emulator -list-avds'. Listed: $Avds"
}

Write-Host ""
Write-Host "Done. Launch the emulator with:" -ForegroundColor Green
Write-Host "  & `"$Emulator`" -avd $AvdName -no-snapshot -grpc 8554"
