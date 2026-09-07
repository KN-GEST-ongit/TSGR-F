[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$DatasetPackageDir = "",

    [Parameter(Mandatory = $false)]
    [string]$WorkspaceRoot = "",

    [ValidateSet(
        "dataset",
        "processing",
        "ground_truth",
        "folds",
        "models",
        "routing",
        "acceptance",
        "evaluation",
        "bootstrap",
        "benchmark",
        "plcc",
        "all"
    )]
    [string]$ThroughStage = "all",

    [int]$MediaPipeWorkers = 16,
    [int]$FoldWorkers = 8,
    [int]$ModelWorkers = 4,
    [int]$BootstrapSamples = 5000,
    [int]$BootstrapSeed = 2026,
    [int]$BenchmarkMaxFramesPerFold = 1000,
    [int]$BenchmarkBatchSize = 64,
    [int]$BenchmarkRepeats = 7,
    [int]$BenchmarkWarmup = 2,
    [int]$PlccModelWorkers = 8,
    [int]$PlccEvaluationWorkers = 8,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ==="
}

function Require-File {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Description`: $Path"
    }
}

function Require-Directory {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing $Description`: $Path"
    }
}

function Test-Checkpoint {
    param([string]$Path)
    return (Test-Path -LiteralPath $Path -PathType Leaf)
}

function Remove-StageOutput {
    param([string]$Path)
    if ($Force -and (Test-Path -LiteralPath $Path)) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Assert-StageOutputReadyOrAbsent {
    param([string]$Directory, [string]$Checkpoint)
    if ((Test-Path -LiteralPath $Directory) -and -not (Test-Checkpoint $Checkpoint)) {
        throw "Incomplete stage output exists at $Directory. Re-run with -Force to rebuild it."
    }
}


function Test-DatasetRuntimeArtifacts {
    param([string]$DatasetRoot)
    if (Test-Path -LiteralPath (Join-Path $DatasetRoot "reports") -PathType Container) {
        return $true
    }
    foreach ($pattern in @(
        "training\*\*\*\runs",
        "testing\*\*\*\*\runs"
    )) {
        $matches = @(Get-ChildItem -Path (Join-Path $DatasetRoot $pattern) -Directory -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($matches.Count -gt 0) {
            return $true
        }
    }
    return $false
}

function Get-ProjectVersion {
    param([string]$ProjectRoot)
    $pyproject = Join-Path $ProjectRoot "pyproject.toml"
    Require-File $pyproject "public pyproject.toml"
    $text = Get-Content -LiteralPath $pyproject -Raw
    $match = [regex]::Match($text, '(?ms)^\[project\].*?^version\s*=\s*"([^"]+)"')
    if (-not $match.Success) {
        throw "Unable to read [project].version from $pyproject"
    }
    return $match.Groups[1].Value
}

function Get-InstalledFrameworkProbe {
    param([string]$PythonPath)
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $null
    }

    # Use ProcessStartInfo instead of direct native invocation. Windows PowerShell 5.1
    # can promote stderr from a failing Python probe to a terminating NativeCommandError
    # when $ErrorActionPreference is Stop, which would prevent stale environments from
    # being detected and rebuilt.
    $probeCode = @'
import importlib.metadata as md
import tsgr
print(md.version("tsgr-framework"))
print(tsgr.__version__)
print(tsgr.__file__)
'@
    $probePath = [System.IO.Path]::Combine(
        [System.IO.Path]::GetTempPath(),
        "tsgr_reviewer_probe_$([guid]::NewGuid().ToString('N')).py"
    )

    try {
        [System.IO.File]::WriteAllText($probePath, $probeCode, [System.Text.UTF8Encoding]::new($false))
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $PythonPath
        $startInfo.Arguments = '"' + $probePath + '"'
        $startInfo.UseShellExecute = $false
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $startInfo.CreateNoWindow = $true

        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $startInfo
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEnd()
        [void]$process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) {
            return $null
        }

        $output = @($stdout -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($output.Count -lt 3) {
            return $null
        }
        return [PSCustomObject]@{
            DistributionVersion = ("$($output[0])").Trim()
            RuntimeVersion = ("$($output[1])").Trim()
            ModulePath = ("$($output[2])").Trim()
        }
    } catch {
        return $null
    } finally {
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    }
}

function Should-StopAfter {
    param([string]$Stage)
    $order = @(
        "dataset",
        "processing",
        "ground_truth",
        "folds",
        "models",
        "routing",
        "acceptance",
        "evaluation",
        "bootstrap",
        "benchmark",
        "plcc"
    )
    if ($ThroughStage -eq "all") {
        return $false
    }
    $requested = [Array]::IndexOf($order, $ThroughStage)
    $current = [Array]::IndexOf($order, $Stage)
    return ($current -ge $requested)
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExpectedFrameworkVersion = Get-ProjectVersion $RepoRoot
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = Join-Path $RepoRoot "reproduction_workspace"
}
$WorkspaceRoot = [System.IO.Path]::GetFullPath($WorkspaceRoot)

if (-not [string]::IsNullOrWhiteSpace($DatasetPackageDir)) {
    $DatasetPackageDir = [System.IO.Path]::GetFullPath($DatasetPackageDir)
}

$DataRoot = Join-Path $WorkspaceRoot "data"
$Dataset = Join-Path $DataRoot "tsgr_dataset"
$Results = Join-Path $WorkspaceRoot "results"
$Models = Join-Path $WorkspaceRoot "models"
$Model = Join-Path $Models "hand_landmarker.task"
$Venv = Join-Path $WorkspaceRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$ScriptsDir = Join-Path $Venv "Scripts"

$TrainingReport = Join-Path $Dataset "reports\training_processing\reference_training"
$ImageReport = Join-Path $Dataset "reports\test_processing\image\high_recall\ignore_handedness\reference_image"
$VideoReport = Join-Path $Dataset "reports\test_processing\video\balanced\recovery_on\ignore_handedness\reference_video"
$HandPresence = Join-Path $Results "ground_truth\hand_presence"
$GroundTruth = Join-Path $Results "ground_truth\final"
$Plan = Join-Path $Results "experiment_plan"
$Fixed = Join-Path $Results "fixed_routing_models"
$Acceptance = Join-Path $Results "acceptance"
$RoutingImage = Join-Path $Results "routing_IMAGE"
$RoutingVideo = Join-Path $Results "routing_VIDEO"
$E2EImage = Join-Path $Results "end_to_end_IMAGE"
$E2EVideo = Join-Path $Results "end_to_end_VIDEO"
$Bootstrap = Join-Path $Results "bootstrap"
$RuntimeImage = Join-Path $Results "runtime_IMAGE"
$RuntimeVideo = Join-Path $Results "runtime_VIDEO"
$Plcc = Join-Path $Results "plcc_sensitivity"
$Branch = "raw.wrist_middle_mcp"

New-Item -ItemType Directory -Force $WorkspaceRoot, $DataRoot, $Results, $Models | Out-Null
$Transcript = Join-Path $Results "reviewer_reproduction.log"
try {
    Start-Transcript -Path $Transcript -Append | Out-Null
} catch {
    Write-Warning "Unable to start transcript: $($_.Exception.Message)"
}

try {
    Write-Step "Reference environment"
    $launcherVersion = (& py -3.12 --version 2>&1 | Out-String).Trim()
    if ($launcherVersion -ne "Python 3.12.10") {
        throw "Reference reproduction requires Python 3.12.10; found: $launcherVersion"
    }
    Write-Host $launcherVersion

    if ($Force -and (Test-Path -LiteralPath $Venv)) {
        Remove-Item -LiteralPath $Venv -Recurse -Force
    }

    if (Test-Path -LiteralPath $Python -PathType Leaf) {
        $existingProbe = Get-InstalledFrameworkProbe $Python
        if ($null -eq $existingProbe -or
            $existingProbe.DistributionVersion -ne $ExpectedFrameworkVersion -or
            $existingProbe.RuntimeVersion -ne $ExpectedFrameworkVersion) {
            Write-Host "Existing reviewer environment does not match public source version $ExpectedFrameworkVersion. Recreating only .venv; data/results are preserved."
            if ($null -ne $existingProbe) {
                Write-Host "Existing distribution version: $($existingProbe.DistributionVersion)"
                Write-Host "Existing runtime version:      $($existingProbe.RuntimeVersion)"
                Write-Host "Existing module path:          $($existingProbe.ModulePath)"
            }
            Remove-Item -LiteralPath $Venv -Recurse -Force
        }
    }

    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        & py -3.12 -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "Failed to create reviewer virtual environment." }
    }

    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
    & $Python -m pip install $RepoRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to install TSGR-F from the public repository." }
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check failed." }

    $installedProbe = Get-InstalledFrameworkProbe $Python
    if ($null -eq $installedProbe) {
        throw "Unable to import the installed TSGR-F package after installation."
    }
    Write-Host "TSGR-F distribution version: $($installedProbe.DistributionVersion)"
    Write-Host "TSGR-F runtime version:      $($installedProbe.RuntimeVersion)"
    Write-Host "TSGR-F module path:          $($installedProbe.ModulePath)"
    if ($installedProbe.DistributionVersion -ne $ExpectedFrameworkVersion -or
        $installedProbe.RuntimeVersion -ne $ExpectedFrameworkVersion) {
        throw "Installed package/runtime version mismatch. Expected $ExpectedFrameworkVersion."
    }
    $venvPrefix = [System.IO.Path]::GetFullPath($Venv).TrimEnd('\') + '\'
    $modulePath = [System.IO.Path]::GetFullPath($installedProbe.ModulePath)
    if (-not $modulePath.StartsWith($venvPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "TSGR-F resolved outside the reviewer virtual environment: $modulePath"
    }
    & $Python -m pip freeze | Set-Content (Join-Path $Results "reviewer_environment.txt")

    $AuditDataset = Join-Path $ScriptsDir "tsgr-audit-dataset.exe"
    $DownloadDataset = Join-Path $ScriptsDir "tsgr-download-dataset.exe"
    $DownloadModel = Join-Path $ScriptsDir "tsgr-download-model.exe"
    $ProcessTraining = Join-Path $ScriptsDir "tsgr-process-training.exe"
    $ProcessTesting = Join-Path $ScriptsDir "tsgr-process-testing.exe"
    $BuildHandPresence = Join-Path $ScriptsDir "tsgr-build-hand-presence-reference.exe"
    $AnalyzeGroundTruth = Join-Path $ScriptsDir "tsgr-analyze-test-ground-truth.exe"
    $GenerateFolds = Join-Path $ScriptsDir "tsgr-generate-folds.exe"
    $BuildFoldModels = Join-Path $ScriptsDir "tsgr-build-fold-models.exe"
    $BuildFixedRouting = Join-Path $ScriptsDir "tsgr-build-fixed-routing.exe"
    $BuildAcceptance = Join-Path $ScriptsDir "tsgr-build-fixed-acceptance.exe"
    $EvaluateRouting = Join-Path $ScriptsDir "tsgr-evaluate-fixed-routing.exe"
    $EvaluateE2E = Join-Path $ScriptsDir "tsgr-evaluate-fixed-end-to-end.exe"
    $BootstrapE2E = Join-Path $ScriptsDir "tsgr-bootstrap-fixed-end-to-end.exe"
    $BenchmarkRouting = Join-Path $ScriptsDir "tsgr-benchmark-fixed-routing.exe"
    $RunPlccSensitivity = Join-Path $ScriptsDir "tsgr-run-plcc-sensitivity.exe"

    foreach ($tool in @(
        $AuditDataset, $DownloadDataset, $DownloadModel, $ProcessTraining, $ProcessTesting,
        $BuildHandPresence, $AnalyzeGroundTruth, $GenerateFolds, $BuildFoldModels,
        $BuildFixedRouting, $BuildAcceptance, $EvaluateRouting, $EvaluateE2E,
        $BootstrapE2E, $BenchmarkRouting, $RunPlccSensitivity
    )) {
        Require-File $tool "installed TSGR-F command"
    }

    $auditHelp = (& $AuditDataset --help 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0 -or $auditHelp -notmatch '--no-release-layout') {
        throw "Installed tsgr-audit-dataset command does not expose --no-release-layout. The reviewer environment is inconsistent with the public source."
    }

    Write-Step "Public dataset"
    if ($Force -and (Test-Path -LiteralPath $Dataset)) {
        Remove-Item -LiteralPath $Dataset -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Dataset "annotations.csv") -PathType Leaf)) {
        if ([string]::IsNullOrWhiteSpace($DatasetPackageDir)) {
            throw "Dataset is not assembled. Pass -DatasetPackageDir with the six public dataset ZIP files."
        }
        Require-Directory $DatasetPackageDir "dataset package directory"
        foreach ($name in @(
            "tsgr_dataset_v1.0__manifest.zip",
            "tsgr_dataset_v1.0__P01.zip",
            "tsgr_dataset_v1.0__P02.zip",
            "tsgr_dataset_v1.0__P03.zip",
            "tsgr_dataset_v1.0__P04.zip",
            "tsgr_dataset_v1.0__P05.zip"
        )) {
            Require-File (Join-Path $DatasetPackageDir $name) "dataset release asset"
        }
        & $DownloadDataset --archive-dir $DatasetPackageDir --output-dir $Dataset --version 1.0 --validate
        if ($LASTEXITCODE -ne 0) { throw "Dataset assembly failed." }
    }
    $runtimeArtifactsPresent = Test-DatasetRuntimeArtifacts $Dataset
    if ($runtimeArtifactsPresent) {
        Write-Host "Auditing public dataset core in working-tree mode (derived runs/reports are allowed)."
        & $AuditDataset $Dataset --require-videos --no-release-layout --strict
        if ($LASTEXITCODE -ne 0) { throw "Working public dataset audit failed." }
    } else {
        Write-Host "Auditing pristine public dataset release layout."
        & $AuditDataset $Dataset --require-videos --release-layout --strict
        if ($LASTEXITCODE -ne 0) { throw "Strict public dataset release audit failed." }
    }
    if (Should-StopAfter "dataset") { return }

    Write-Step "MediaPipe model"
    if ($Force -and (Test-Path -LiteralPath $Model)) {
        Remove-Item -LiteralPath $Model -Force
    }
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
        & $DownloadModel --output $Model
        if ($LASTEXITCODE -ne 0) { throw "MediaPipe model download failed." }
    }
    $modelHash = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Model SHA-256: $modelHash"

    Write-Step "Training processing"
    $trainingCheckpoint = Join-Path $TrainingReport "processing_report.json"
    Remove-StageOutput $TrainingReport
    Assert-StageOutputReadyOrAbsent $TrainingReport $trainingCheckpoint
    if (-not (Test-Checkpoint $trainingCheckpoint)) {
        & $ProcessTraining $Dataset --model $Model --all-data --detection-profile high_recall --report-name reference_training --workers $MediaPipeWorkers --progress
        if ($LASTEXITCODE -ne 0) { throw "Training processing failed." }
    } else {
        Write-Host "Reusing completed training processing: $TrainingReport"
    }

    Write-Step "IMAGE processing"
    $imageCheckpoint = Join-Path $ImageReport "processing_report.json"
    Remove-StageOutput $ImageReport
    Assert-StageOutputReadyOrAbsent $ImageReport $imageCheckpoint
    if (-not (Test-Checkpoint $imageCheckpoint)) {
        & $ProcessTesting $Dataset --model $Model --test-mode image --handedness-policy ignore_handedness --all-data --minimum-success-rate 0.95 --detection-profile high_recall --no-video-recovery --report-name reference_image --workers $MediaPipeWorkers --progress --no-fail-fast
        if ($LASTEXITCODE -ne 0) { throw "IMAGE processing failed." }
    } else {
        Write-Host "Reusing completed IMAGE processing: $ImageReport"
    }

    Write-Step "VIDEO processing"
    $videoCheckpoint = Join-Path $VideoReport "processing_report.json"
    Remove-StageOutput $VideoReport
    Assert-StageOutputReadyOrAbsent $VideoReport $videoCheckpoint
    if (-not (Test-Checkpoint $videoCheckpoint)) {
        & $ProcessTesting $Dataset --model $Model --test-mode video --handedness-policy ignore_handedness --all-data --fallback-fps 30 --minimum-success-rate 0.95 --detection-profile balanced --video-recovery --video-recovery-after 1 --report-name reference_video --workers $MediaPipeWorkers --progress --no-fail-fast
        if ($LASTEXITCODE -ne 0) { throw "VIDEO processing failed." }
    } else {
        Write-Host "Reusing completed VIDEO processing: $VideoReport"
    }
    if (Should-StopAfter "processing") { return }

    Write-Step "Hand-presence reference"
    $handCheckpoint = Join-Path $HandPresence "hand_presence_reference.json"
    Remove-StageOutput $HandPresence
    Assert-StageOutputReadyOrAbsent $HandPresence $handCheckpoint
    if (-not (Test-Checkpoint $handCheckpoint)) {
        & $BuildHandPresence $Dataset --model $Model --output-dir $HandPresence --workers $MediaPipeWorkers --progress
        if ($LASTEXITCODE -ne 0) { throw "Hand-presence reference build failed." }
    } else {
        Write-Host "Reusing completed hand-presence reference: $HandPresence"
    }

    Write-Step "Frame-level ground truth"
    $groundTruthCheckpoint = Join-Path $GroundTruth "ground_truth_report.json"
    Remove-StageOutput $GroundTruth
    Assert-StageOutputReadyOrAbsent $GroundTruth $groundTruthCheckpoint
    if (-not (Test-Checkpoint $groundTruthCheckpoint)) {
        & $AnalyzeGroundTruth $Dataset --hand-presence-reference $HandPresence --output-dir $GroundTruth
        if ($LASTEXITCODE -ne 0) { throw "Ground-truth analysis failed." }
    } else {
        Write-Host "Reusing completed ground truth: $GroundTruth"
    }
    if (Should-StopAfter "ground_truth") { return }

    Write-Step "S1-S4 experiment plan"
    $planCheckpoint = Join-Path $Plan "experiment_plan.json"
    Remove-StageOutput $Plan
    Assert-StageOutputReadyOrAbsent $Plan $planCheckpoint
    if (-not (Test-Checkpoint $planCheckpoint)) {
        & $GenerateFolds $Dataset --output-dir $Plan --workers $FoldWorkers --progress
        if ($LASTEXITCODE -ne 0) { throw "Experiment-plan generation failed." }
    } else {
        Write-Host "Reusing experiment plan: $Plan"
    }
    $planPayload = Get-Content -LiteralPath $planCheckpoint -Raw | ConvertFrom-Json
    if ([int]$planPayload.active_fold_count -ne 24) {
        throw "Expected 24 active folds for dataset v1.0; found $($planPayload.active_fold_count)."
    }
    if (Should-StopAfter "folds") { return }

    Write-Step "Fold-local reference models"
    $existingModelSets = @(
        Get-ChildItem -LiteralPath $Plan -Recurse -Filter "model_set.json" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "reference_model_variants" }
    )
    if ($Force) {
        foreach ($path in Get-ChildItem -LiteralPath $Plan -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq "reference_models" }) {
            Remove-Item -LiteralPath $path.FullName -Recurse -Force
        }
        $existingModelSets = @()
    }
    if ($existingModelSets.Count -lt [int]$planPayload.active_fold_count) {
        & $BuildFoldModels $Plan --all-scenarios --all-folds --branch $Branch --compact-correlation-threshold 0.995 --workers $ModelWorkers --progress
        if ($LASTEXITCODE -ne 0) { throw "Fold-local reference-model build failed." }
        $existingModelSets = @(
            Get-ChildItem -LiteralPath $Plan -Recurse -Filter "model_set.json" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch "reference_model_variants" }
        )
    } else {
        Write-Host "Reusing $($existingModelSets.Count) fold-local reference model sets."
    }
    if ($existingModelSets.Count -lt [int]$planPayload.active_fold_count) {
        throw "Fold-local reference-model build is incomplete: $($existingModelSets.Count)/$($planPayload.active_fold_count)."
    }
    if (Should-StopAfter "models") { return }

    Write-Step "Fixed routing models"
    $fixedCheckpoint = Join-Path $Fixed "frozen_model_build_report.json"
    Remove-StageOutput $Fixed
    Assert-StageOutputReadyOrAbsent $Fixed $fixedCheckpoint
    if (-not (Test-Checkpoint $fixedCheckpoint)) {
        & $BuildFixedRouting $Plan --dataset-root-override $Dataset --output-dir $Fixed --all-scenarios --branch $Branch --feature-set compact --orientation-mode camera_aware --os-alpha 0.65 --c-top-n 5 --progress
        if ($LASTEXITCODE -ne 0) { throw "Fixed routing build failed." }
    } else {
        Write-Host "Reusing fixed routing models: $Fixed"
    }
    if (Should-StopAfter "routing") { return }

    Write-Step "TRAIN-only acceptance"
    $acceptanceCheckpoint = Join-Path $Acceptance "frozen_acceptance_build_report.json"
    Remove-StageOutput $Acceptance
    Assert-StageOutputReadyOrAbsent $Acceptance $acceptanceCheckpoint
    if (-not (Test-Checkpoint $acceptanceCheckpoint)) {
        & $BuildAcceptance $Plan --fixed-model-dir $Fixed --output-dir $Acceptance --all-scenarios --objective mcc --min-positive-recall 0.99 --progress
        if ($LASTEXITCODE -ne 0) { throw "Acceptance calibration failed." }
    } else {
        Write-Host "Reusing acceptance calibration: $Acceptance"
    }
    if (Should-StopAfter "acceptance") { return }

    Write-Step "IMAGE routing evaluation"
    $routingImageCheckpoint = Join-Path $RoutingImage "scenario_summary.csv"
    Remove-StageOutput $RoutingImage
    Assert-StageOutputReadyOrAbsent $RoutingImage $routingImageCheckpoint
    if (-not (Test-Checkpoint $routingImageCheckpoint)) {
        & $EvaluateRouting $Plan --dataset-root-override $Dataset --fixed-model-dir $Fixed --test-processing-report $ImageReport --ground-truth-report $GroundTruth --output-dir $RoutingImage --all-scenarios --branch $Branch --no-copy-problem-cases --progress
        if ($LASTEXITCODE -ne 0) { throw "IMAGE routing evaluation failed." }
    }

    Write-Step "VIDEO routing evaluation"
    $routingVideoCheckpoint = Join-Path $RoutingVideo "scenario_summary.csv"
    Remove-StageOutput $RoutingVideo
    Assert-StageOutputReadyOrAbsent $RoutingVideo $routingVideoCheckpoint
    if (-not (Test-Checkpoint $routingVideoCheckpoint)) {
        & $EvaluateRouting $Plan --dataset-root-override $Dataset --fixed-model-dir $Fixed --test-processing-report $VideoReport --ground-truth-report $GroundTruth --output-dir $RoutingVideo --all-scenarios --branch $Branch --no-copy-problem-cases --progress
        if ($LASTEXITCODE -ne 0) { throw "VIDEO routing evaluation failed." }
    }

    Write-Step "IMAGE end-to-end evaluation"
    $e2eImageCheckpoint = Join-Path $E2EImage "scenario_summary.csv"
    Remove-StageOutput $E2EImage
    Assert-StageOutputReadyOrAbsent $E2EImage $e2eImageCheckpoint
    if (-not (Test-Checkpoint $e2eImageCheckpoint)) {
        & $EvaluateE2E $Plan --dataset-root-override $Dataset --fixed-model-dir $Fixed --acceptance-dir $Acceptance --test-processing-report $ImageReport --ground-truth-report $GroundTruth --output-dir $E2EImage --branch $Branch --all-scenarios --sequence-max-lag-frames 10 --prediction-batch-size 64 --no-copy-problem-cases --progress
        if ($LASTEXITCODE -ne 0) { throw "IMAGE end-to-end evaluation failed." }
    }

    Write-Step "VIDEO end-to-end evaluation"
    $e2eVideoCheckpoint = Join-Path $E2EVideo "scenario_summary.csv"
    Remove-StageOutput $E2EVideo
    Assert-StageOutputReadyOrAbsent $E2EVideo $e2eVideoCheckpoint
    if (-not (Test-Checkpoint $e2eVideoCheckpoint)) {
        & $EvaluateE2E $Plan --dataset-root-override $Dataset --fixed-model-dir $Fixed --acceptance-dir $Acceptance --test-processing-report $VideoReport --ground-truth-report $GroundTruth --output-dir $E2EVideo --branch $Branch --all-scenarios --sequence-max-lag-frames 10 --prediction-batch-size 64 --no-copy-problem-cases --progress
        if ($LASTEXITCODE -ne 0) { throw "VIDEO end-to-end evaluation failed." }
    }
    if (Should-StopAfter "evaluation") { return }

    Write-Step "Clustered bootstrap (take-level)"
    $bootstrapCheckpoint = Join-Path $Bootstrap "frozen_end_to_end_bootstrap_report.json"
    Remove-StageOutput $Bootstrap
    Assert-StageOutputReadyOrAbsent $Bootstrap $bootstrapCheckpoint
    if (-not (Test-Checkpoint $bootstrapCheckpoint)) {
        & $BootstrapE2E --image-report $E2EImage --video-report $E2EVideo --output-dir $Bootstrap --samples $BootstrapSamples --seed $BootstrapSeed
        if ($LASTEXITCODE -ne 0) { throw "Clustered bootstrap failed." }
    } else {
        Write-Host "Reusing clustered bootstrap: $Bootstrap"
    }
    if (Should-StopAfter "bootstrap") { return }

    Write-Step "IMAGE fixed-routing runtime benchmark"
    $runtimeImageCheckpoint = Join-Path $RuntimeImage "frozen_routing_runtime_report.json"
    Remove-StageOutput $RuntimeImage
    Assert-StageOutputReadyOrAbsent $RuntimeImage $runtimeImageCheckpoint
    if (-not (Test-Checkpoint $runtimeImageCheckpoint)) {
        & $BenchmarkRouting $Plan --dataset-root-override $Dataset --fixed-model-dir $Fixed --test-processing-report $ImageReport --output-dir $RuntimeImage --branch $Branch --all-scenarios --max-frames-per-fold $BenchmarkMaxFramesPerFold --batch-size $BenchmarkBatchSize --repeats $BenchmarkRepeats --warmup $BenchmarkWarmup --progress
        if ($LASTEXITCODE -ne 0) { throw "IMAGE routing benchmark failed." }
    } else {
        Write-Host "Reusing IMAGE runtime benchmark: $RuntimeImage"
    }

    Write-Step "VIDEO fixed-routing runtime benchmark"
    $runtimeVideoCheckpoint = Join-Path $RuntimeVideo "frozen_routing_runtime_report.json"
    Remove-StageOutput $RuntimeVideo
    Assert-StageOutputReadyOrAbsent $RuntimeVideo $runtimeVideoCheckpoint
    if (-not (Test-Checkpoint $runtimeVideoCheckpoint)) {
        & $BenchmarkRouting $Plan --dataset-root-override $Dataset --fixed-model-dir $Fixed --test-processing-report $VideoReport --output-dir $RuntimeVideo --branch $Branch --all-scenarios --max-frames-per-fold $BenchmarkMaxFramesPerFold --batch-size $BenchmarkBatchSize --repeats $BenchmarkRepeats --warmup $BenchmarkWarmup --progress
        if ($LASTEXITCODE -ne 0) { throw "VIDEO routing benchmark failed." }
    } else {
        Write-Host "Reusing VIDEO runtime benchmark: $RuntimeVideo"
    }
    if (Should-StopAfter "benchmark") { return }

    Write-Step "Post-hoc PLCC sensitivity"
    $plccCheckpoint = Join-Path $Plcc "frozen_plcc_sensitivity_report.json"
    if ($Force -and (Test-Path -LiteralPath $Plcc)) {
        Remove-Item -LiteralPath $Plcc -Recurse -Force
    }
    if (-not (Test-Checkpoint $plccCheckpoint)) {
        & $RunPlccSensitivity $Plan --dataset-root-override $Dataset --test-processing-report $ImageReport --ground-truth-report $GroundTruth --output-dir $Plcc --branch $Branch --scenario S1_ALL_IN_DOMAIN --scenario S2_LOBO --scenario S3_LOSO --scenario S4_LOSO_BACKGROUND --standard-grid --model-workers $PlccModelWorkers --evaluation-workers $PlccEvaluationWorkers --evaluation-batch-size 64 --resume --progress
        if ($LASTEXITCODE -ne 0) { throw "PLCC sensitivity analysis failed." }
    } else {
        Write-Host "Reusing PLCC sensitivity analysis: $Plcc"
    }
    if (Should-StopAfter "plcc") { return }

    $handPayload = Get-Content -LiteralPath (Join-Path $HandPresence "hand_presence_reference.json") -Raw | ConvertFrom-Json
    $summary = [ordered]@{
        schema_version = "tsgr_reviewer_reproduction_v1"
        tsgr_version = $installedProbe.RuntimeVersion
        python = (& $Python --version 2>&1 | Out-String).Trim()
        workspace_root = "."
        dataset_root = "data/tsgr_dataset"
        model_sha256 = $modelHash
        hand_presence_reference_sha256 = $handPayload.reference_sha256
        active_fold_count = [int]$planPayload.active_fold_count
        fixed_routing_complete = (Test-Checkpoint $fixedCheckpoint)
        acceptance_complete = (Test-Checkpoint $acceptanceCheckpoint)
        routing_image_complete = (Test-Checkpoint $routingImageCheckpoint)
        routing_video_complete = (Test-Checkpoint $routingVideoCheckpoint)
        end_to_end_image_complete = (Test-Checkpoint $e2eImageCheckpoint)
        end_to_end_video_complete = (Test-Checkpoint $e2eVideoCheckpoint)
        bootstrap_samples = $BootstrapSamples
        bootstrap_seed = $BootstrapSeed
        bootstrap_complete = (Test-Checkpoint $bootstrapCheckpoint)
        runtime_image_complete = (Test-Checkpoint $runtimeImageCheckpoint)
        runtime_video_complete = (Test-Checkpoint $runtimeVideoCheckpoint)
        plcc_complete = (Test-Checkpoint $plccCheckpoint)
        plcc_post_hoc_exploratory = $true
        plcc_eligible_for_final_model_reselection = $false
    }
    $summary | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Results "reviewer_reproduction_report.json")

    Write-Step "Reference reproduction complete"
    Write-Host "Results: $Results"
    Write-Host "Run report: $(Join-Path $Results 'reviewer_reproduction_report.json')"
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}
