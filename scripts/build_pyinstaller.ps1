param(
    [string]$Python = "python",
    [string]$OutDir = "dist\pyinstaller",
    [string]$Name = "pcwannier",
    [switch]$OneFile,
    [switch]$WithNumba,
    [switch]$Clean,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
Set-Location $repoRoot

function Resolve-RepositoryPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Test-StrictDescendant([string]$Candidate, [string]$Root) {
    $prefix = $Root.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    return $Candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

$OutDir = Resolve-RepositoryPath $OutDir
$cleanRoots = @(
    (Resolve-RepositoryPath "dist"),
    (Resolve-RepositoryPath "build")
)
if ($Clean -and -not ($cleanRoots | Where-Object { Test-StrictDescendant $OutDir $_ })) {
    throw "Refusing to clean '$OutDir'. Clean targets must be strict descendants of '$repoRoot\dist' or '$repoRoot\build'."
}

$entry = Join-Path $repoRoot "scripts\pyinstaller_entry.py"
$workDir = Resolve-RepositoryPath "build\pyinstaller"

$pyinstallerOptions = @(
    "-m", "PyInstaller",
    "--clean",
    "--noconfirm",
    "--name", $Name,
    "--console",
    "--distpath", $OutDir,
    "--workpath", $workDir,
    "--specpath", $workDir,
    "--collect-submodules", "scipy",
    "--collect-data", "matplotlib",
    "--collect-data", "pcwannier.symmetry"
)

if ($OneFile) {
    $pyinstallerOptions = $pyinstallerOptions + @("--onefile")
}

if ($WithNumba) {
    $pyinstallerOptions = $pyinstallerOptions + @(
        "--collect-submodules", "numba",
        "--collect-submodules", "llvmlite"
    )
}

$argsList = $pyinstallerOptions + @($entry)

Write-Host "Repository: $repoRoot"
Write-Host "Python: $Python"
Write-Host "Output: $OutDir"
Write-Host "Command: $Python $($argsList -join ' ')"
if ($Clean) {
    Write-Host "Clean target: $OutDir"
}

if ($DryRun) {
    Write-Host "Dry run only; PyInstaller was not executed."
    exit 0
}

if ($Clean -and (Test-Path -LiteralPath $OutDir)) {
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$timer = [Diagnostics.Stopwatch]::StartNew()
& $Python @argsList
$timer.Stop()

$exe = Get-ChildItem -Path $OutDir -Recurse -Filter "$Name.exe" | Select-Object -First 1

Write-Host ""
if ($exe) {
    Write-Host "Build complete: $($exe.FullName)"
} else {
    Write-Host "Build complete. Check output directory: $OutDir"
}
Write-Host ("Build time: {0:hh\:mm\:ss\.fff}" -f $timer.Elapsed)
