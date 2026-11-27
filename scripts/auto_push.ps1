[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Watch,
    [switch]$RunOnce
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ExpectedBranch = 'main'
$TaskName = 'GestureAI-Auto-Push'
$LogDirectory = Join-Path $env:LOCALAPPDATA 'GestureAI'
$LogPath = Join-Path $LogDirectory 'auto-push.log'

function Write-Log {
    param([Parameter(Mandatory)][string]$Message)

    if (-not (Test-Path -LiteralPath $LogDirectory)) {
        New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    }
    $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

function Invoke-Git {
    param([Parameter(Mandatory)][string[]]$GitArgs)

    $output = & git -C $RepoRoot @GitArgs 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw ('Git command failed (exit {0}): git {1}' -f $exitCode, ($GitArgs -join ' '))
    }
    return $output
}

function Assert-SyncTarget {
    $insideRepo = (Invoke-Git -GitArgs @('rev-parse', '--is-inside-work-tree') | Select-Object -First 1).ToString().Trim()
    if ($insideRepo -ne 'true') {
        throw 'This folder is not a Git repository.'
    }

    $branch = (Invoke-Git -GitArgs @('branch', '--show-current') | Select-Object -First 1).ToString().Trim()
    if ($branch -ne $ExpectedBranch) {
        throw ('Auto-push is limited to branch {0}; current branch is {1}.' -f $ExpectedBranch, $branch)
    }

    $remote = (Invoke-Git -GitArgs @('remote', 'get-url', 'origin') | Select-Object -First 1).ToString().Trim()
    $remoteUri = [System.Uri]$remote
    if ($remoteUri.Scheme -ne 'https' -or $remoteUri.Host -ne 'github.com' -or $remoteUri.UserInfo) {
        throw 'The origin must be a clean HTTPS GitHub URL with no embedded username or token.'
    }

    $userName = (Invoke-Git -GitArgs @('config', 'user.name') | Select-Object -First 1)
    $userEmail = (Invoke-Git -GitArgs @('config', 'user.email') | Select-Object -First 1)
    if (-not $userName -or -not $userEmail) {
        throw 'Git commit name or email is not configured.'
    }
}

function Get-WorkingSignature {
    $items = & git -C $RepoRoot status --porcelain --untracked-files=all 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not read the repository status.'
    }
    return [string]::Join("`n", [string[]]$items)
}

function Test-StagedFilesForSecrets {
    $paths = @(Invoke-Git -GitArgs @('diff', '--cached', '--name-only', '--diff-filter=ACMRT'))
    $textExtensions = @('.py', '.ps1', '.md', '.txt', '.toml', '.json', '.yaml', '.yml', '.ini', '.cfg', '.bat', '.cmd', '.sh', '.js', '.html', '.css', '.xml', '.csv')
    $secretPattern = '(?i)(github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|freellmapi-[A-Za-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*["'']?[A-Za-z0-9._/+=-]{24,})'

    foreach ($path in $paths) {
        $extension = [System.IO.Path]::GetExtension($path).ToLowerInvariant()
        if ($extension -notin $textExtensions) {
            continue
        }

        $contentLines = & git -C $RepoRoot show (':{0}' -f $path) 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw ('Could not inspect staged file: {0}' -f $path)
        }
        $content = [string]::Join("`n", [string[]]$contentLines)
        if ($content -match $secretPattern) {
            throw ('Possible credential found in staged file: {0}. Auto-push stopped; remove the secret before syncing.' -f $path)
        }
    }
}

function Sync-Repository {
    Assert-SyncTarget

    Invoke-Git -GitArgs @('add', '-A') | Out-Null
    # Remove previously tracked Python bytecode from the published index while
    # leaving local cache files untouched. Do this after staging so Git does not
    # immediately add the tracked cache files back.
    Invoke-Git -GitArgs @('rm', '-r', '--cached', '--ignore-unmatch', '--', 'app/__pycache__', 'app/camera/__pycache__', 'app/ui/__pycache__') | Out-Null
    $staged = @(Invoke-Git -GitArgs @('diff', '--cached', '--name-only'))

    if ($staged.Count -gt 0) {
        Test-StagedFilesForSecrets
        Invoke-Git -GitArgs @('diff', '--cached', '--check') | Out-Null

        $message = 'Auto sync Gesture.AI {0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm')
        Invoke-Git -GitArgs @('commit', '-m', $message) | Out-Null
        Write-Log ('Created a commit with {0} staged file changes.' -f $staged.Count)
    }

    $pushOutput = & git -C $RepoRoot push --set-upstream origin $ExpectedBranch 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Log 'Push did not complete. Check GitHub Desktop sign-in and the repository status; it will retry later.'
        return $false
    }

    Write-Log 'Repository sync completed successfully.'
    return $true
}

function Install-AutoPushTask {
    Assert-SyncTarget

    $scriptPath = Join-Path $PSScriptRoot 'auto_push.ps1'
    $powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $arguments = '-NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -Watch' -f $scriptPath
    $userId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

    $action = New-ScheduledTaskAction -Execute $powershellPath -Argument $arguments -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
    $principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Waits for Gesture.AI file changes, then safely commits and pushes them to the existing GitHub master branch.' -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    Write-Log 'Automatic sync was enabled at Windows sign-in.'
    Write-Host 'Automatic GitHub sync is enabled. It runs hidden after sign-in and waits for changes to settle before syncing.'
}

if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host 'Automatic GitHub sync has been turned off.'
    } else {
        Write-Host 'Automatic GitHub sync was not installed.'
    }
    exit 0
}

if ($Install) {
    Install-AutoPushTask
    exit 0
}

if ($RunOnce) {
    if (-not (Sync-Repository)) {
        Write-Host 'Sync could not finish. Sign in to GitHub Desktop, then run this script again.'
        exit 1
    }
    Write-Host 'Project synced to GitHub.'
    exit 0
}

if ($Watch) {
    try {
        Assert-SyncTarget
        Write-Log 'Auto-sync watcher started.'
        $signature = Get-WorkingSignature
        $changedAt = if ([string]::IsNullOrWhiteSpace($signature)) { Get-Date } else { (Get-Date).AddSeconds(-25) }
        $lastPushAttempt = [datetime]::MinValue
        # A clean working tree can still have local commits that were not pushed
        # (for example, after a temporary sign-in or network failure).
        $pushPending = $true

        if (Sync-Repository) {
            $pushPending = $false
        } else {
            $lastPushAttempt = Get-Date
        }

        while ($true) {
            Start-Sleep -Seconds 5
            $currentSignature = Get-WorkingSignature
            if ($currentSignature -ne $signature) {
                $signature = $currentSignature
                $changedAt = Get-Date
            }

            $stableChangeReady = -not [string]::IsNullOrWhiteSpace($signature) -and ((Get-Date) - $changedAt).TotalSeconds -ge 25
            $retryReady = $pushPending -and ((Get-Date) - $lastPushAttempt).TotalSeconds -ge 60

            if ($stableChangeReady -or $retryReady) {
                $lastPushAttempt = Get-Date
                $pushPending = -not (Sync-Repository)
                $signature = Get-WorkingSignature
                $changedAt = Get-Date
            }
        }
    }
    catch {
        Write-Log ('Auto-sync stopped: {0}' -f $_.Exception.Message)
        exit 1
    }
}

Write-Host 'Use -Install to enable background auto-push, -RunOnce to sync now, or -Uninstall to turn it off.'
