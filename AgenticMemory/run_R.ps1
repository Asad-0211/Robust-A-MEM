# Series R: rebuild with each build-time variant (own cache folder per variant), then answer.
# Usage:  .\run_R.ps1                                   all runs, conversation 1
#         .\run_R.ps1 -Only R0_control,R1_noevo,R4_validate_only -Samples "0,8"
param([string[]]$Only = @(), [string]$Samples = "1")
$sfx = if ($Samples -eq "1") { "" } else { "_s" + ($Samples -replace ",", "-") }

$runs = @(
    @{ name = "R0_control";         v = "bt0,at0" },
    @{ name = "R1_noevo";           v = "bt0,noevo,at0" },
    @{ name = "R2_noupdate";        v = "bt0,noupdate,at0" },
    @{ name = "R3_harden_validate"; v = "bt0,harden,validate,history,at0" },
    @{ name = "R4_validate_only";   v = "bt0,validate,history,at0" }
)

foreach ($r in $runs) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $r.name)) { continue }
    Write-Host ""
    Write-Host "=== $($r.name)  samples=$Samples  AMEM_VARIANT=$($r.v)  $(Get-Date -Format 'HH:mm:ss') ==="
    $env:AMEM_VARIANT = $r.v
    python test_v2.py --backend ollama --model qwen2.5-3b --dataset data/locomo10.json --samples $Samples --retrieve_k 10 --legacy_c5_reference --output "results_$($r.name)$sfx.json" 2>&1 | Tee-Object -FilePath "run_$($r.name)$sfx.txt"
}
$env:AMEM_VARIANT = ""
Write-Host "Series R finished $(Get-Date -Format 'HH:mm:ss')"
