param(
    [string]$Python = "python",
    [string]$OutDir = "dist\nuitka",
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

$entry = Join-Path $repoRoot "scripts\nuitka_entry.py"

$nuitkaOptions = @(
    "-m", "nuitka",
    "--standalone",
    "--assume-yes-for-downloads",
    "--enable-plugin=matplotlib",
    "--include-package=pcwannier",
    "--include-package-data=pcwannier.symmetry",
    "--output-dir=$OutDir",
    "--output-filename=$Name.exe"
)

if ($OneFile) {
    $nuitkaOptions = @("-m", "nuitka", "--onefile") + $nuitkaOptions[2..($nuitkaOptions.Count - 1)]
}

if ($WithNumba) {
    $nuitkaOptions = $nuitkaOptions + @("--include-package=numba", "--include-package=llvmlite")
}

$argsList = $nuitkaOptions + @($entry)

Write-Host "Repository: $repoRoot"
Write-Host "Python: $Python"
Write-Host "Output: $OutDir"
Write-Host "Command: $Python $($argsList -join ' ')"
if ($Clean) {
    Write-Host "Clean target: $OutDir"
}

if ($DryRun) {
    Write-Host "Dry run only; Nuitka was not executed."
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
