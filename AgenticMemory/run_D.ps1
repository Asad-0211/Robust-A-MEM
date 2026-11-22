# Series D: text repairs on the cached no-evolution notes (read-time, no rebuild).
# Usage:  .\run_D.ps1 -Only D0_control,D2_datefix -Samples "0,1,2,8,9" -Categories "2"
param([string[]]$Only = @(), [string]$Samples = "1", [string]$Categories = "")
$sfx = if ($Samples -eq "1") { "" } else { "_s" + ($Samples -replace ",", "-") }
if ($Categories -ne "") { $sfx = $sfx + "_c" + ($Categories -replace ",", "-") }
$catArg = if ($Categories -ne "") { @("--categories", $Categories) } else { @() }

$runs = @(
    @{ name = "D0_control";  v = "bt0,noevo,reembed,at0" },
    @{ name = "D1_tagfix";   v = "bt0,noevo,reembed,tagfix,at0" },
    @{ name = "D2_datefix";  v = "bt0,noevo,reembed,datefix,at0" },
    @{ name = "D3_both";     v = "bt0,noevo,reembed,tagfix,datefix,at0" }
)

foreach ($r in $runs) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $r.name)) { continue }
    Write-Host ""
    Write-Host "=== $($r.name)  samples=$Samples  categories=$Categories  AMEM_VARIANT=$($r.v)  $(Get-Date -Format 'HH:mm:ss') ==="
    $env:AMEM_VARIANT = $r.v
    python test_v2.py --backend ollama --model qwen2.5-3b --dataset data/locomo10.json --samples $Samples @catArg --retrieve_k 10 --legacy_c5_reference --output "results_$($r.name)$sfx.json" 2>&1 | Tee-Object -FilePath "run_$($r.name)$sfx.txt"
}
$env:AMEM_VARIANT = ""
Write-Host "Series D finished $(Get-Date -Format 'HH:mm:ss')"
