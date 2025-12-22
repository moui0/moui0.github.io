# change deploy config
(Get-Content "config/deploy.yaml") `
    -replace 'KeepLocalChanges:\s*false', 'KeepLocalChanges: true' |
    Set-Content $path -Encoding UTF8

# patch before config_updater
$patchFiles = Get-ChildItem -Path . -Filter *.patch -File |
    Where-Object { -not $_.Name.StartsWith("i18n-") }

foreach ($patch in $patchFiles) {
    Write-Host "Applying patch: $($patch.Name)"
    git apply "$($patch.FullName)"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Apply patch failed: $($patch.Name)"
        exit 1
    }
}

# run config_updater
& ".\toolkit\python.exe" -m module.config.config_updater

if ($LASTEXITCODE -ne 0) {
    Write-Error "config_updater failed"
    exit 1
}

# patch after config_updater
$patchFiles = Get-ChildItem -Path . -Filter "i18n-*.patch" -File | Sort-Object Name

foreach ($patch in $patchFiles) {
    Write-Host "Applying patch: $($patch.Name)"
    git apply "$($patch.FullName)"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Apply patch failed: $($patch.Name)"
        exit 1
    }
}

Write-Host "All patches applied successfully."
