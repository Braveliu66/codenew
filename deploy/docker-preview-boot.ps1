param(
    [ValidateSet("Start", "Save")]
    [string]$Mode = "Start",

    [string]$CacheDir = "Q:\docker-image-cache",
    [string]$ImageArchive = "three-dgs-preview-images.tar"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "deploy\docker-compose.preview.yml"
$ArchivePath = Join-Path $CacheDir $ImageArchive

$Images = @(
    "three-dgs-gpu-runtime:local",
    "three-dgs-api:local",
    "deploy-frontend",
    "postgres:16",
    "redis:7",
    "minio/minio:RELEASE.2025-04-22T22-12-26Z"
)

function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-DockerImage($Image) {
    docker image inspect $Image *> $null
    return $LASTEXITCODE -eq 0
}

function Wait-DockerReady {
    if (-not (Test-Command "docker")) {
        throw "docker command was not found. Install and start Docker Desktop first."
    }

    for ($i = 1; $i -le 60; $i++) {
        docker info *> $null
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds 2
    }

    throw "Docker is not ready. Make sure Docker Desktop is running."
}

Set-Location $ProjectRoot
Wait-DockerReady

if ($Mode -eq "Save") {
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

    Write-Host "Building and starting preview stack..."
    docker compose -f $ComposeFile up -d --build
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up -d --build failed."
    }

    $MissingAfterBuild = @($Images | Where-Object { -not (Test-DockerImage $_) })
    if ($MissingAfterBuild.Count -gt 0) {
        throw "These images are missing and cannot be saved: $($MissingAfterBuild -join ', ')"
    }

    Write-Host "Saving images to $ArchivePath ..."
    docker save -o $ArchivePath $Images
    if ($LASTEXITCODE -ne 0) {
        throw "docker save failed."
    }

    Write-Host "Done. After C drive reset, reinstall Docker Desktop and run:"
    Write-Host "powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit 0
}

$MissingImages = @($Images | Where-Object { -not (Test-DockerImage $_) })
if ($MissingImages.Count -gt 0) {
    if (-not (Test-Path $ArchivePath)) {
        throw "Missing images: $($MissingImages -join ', '). Image archive was not found: $ArchivePath. Run -Mode Save before reset."
    }

    Write-Host "Loading images from $ArchivePath ..."
    docker load -i $ArchivePath
    if ($LASTEXITCODE -ne 0) {
        throw "docker load failed."
    }
}

$StillMissing = @($Images | Where-Object { -not (Test-DockerImage $_) })
if ($StillMissing.Count -gt 0) {
    throw "Images are still missing after load: $($StillMissing -join ', ')"
}

Write-Host "Starting preview stack without rebuilding..."
docker compose -f $ComposeFile up -d --no-build
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up -d --no-build failed."
}

Write-Host "Started. Frontend: http://localhost:3001  Backend: http://localhost:8000"
