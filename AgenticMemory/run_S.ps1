# Series S: read-time variants on cached memories (no rebuild).
# Usage:  .\run_S.ps1                                   all runs, conversation 1
#         .\run_S.ps1 -Only S1_reembed,S3_contentonly -Samples "0,8"
param([string[]]$Only = @(), [string]$Samples = "1")
$sfx = if ($Samples -eq "1") { "" } else { "_s" + ($Samples -replace ",", "-") }

$runs = @(
    @{ name = "S0_control";     v = "at0" },
    @{ name = "S1_reembed";     v = "reembed,at0" },
    @{ name = "S2_blank";       v = "blank,at0" },
    @{ name = "S3_contentonly"; v = "embcontent,ctxmin,at0" },
    @{ name = "S4_dedupe";      v = "reembed,dedupe,at0" },
    @{ name = "S5_dedupe_rev";  v = "reembed,dedupe,rev,at0" },
    @{ name = "S6_qplus";       v = "reembed,qplus,at0" }
)

foreach ($r in $runs) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $r.name)) { continue }
    Write-Host ""
    Write-Host "=== $($r.name)  samples=$Samples  AMEM_VARIANT=$($r.v)  $(Get-Date -Format 'HH:mm:ss') ==="
    $env:AMEM_VARIANT = $r.v
    python test_v2.py --backend ollama --model qwen2.5-3b --dataset data/locomo10.json --samples $Samples --retrieve_k 10 --legacy_c5_reference --output "results_$($r.name)$sfx.json" 2>&1 | Tee-Object -FilePath "run_$($r.name)$sfx.txt"
}
$env:AMEM_VARIANT = ""
Write-Host "Series S finished $(Get-Date -Format 'HH:mm:ss')"
