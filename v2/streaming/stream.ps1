# Sentinel Credit v2, Phase 4: control script for the local streaming stack (V2-03).
#
#   .\v2\streaming\stream.ps1 start                 build if needed, start Kafka, Flink, sink, agent; create topics and schema
#   .\v2\streaming\stream.ps1 submit                submit the PyFlink job (once per fresh cluster)
#   .\v2\streaming\stream.ps1 replay 100 4000       replay 4,000 customers' histories at 100 events/s, then run the checks
#   .\v2\streaming\stream.ps1 status                containers, Flink jobs, recent runs
#   .\v2\streaming\stream.ps1 stop                  stop containers, keep everything (resume with start)
#   .\v2\streaming\stream.ps1 teardown              remove containers, Kafka data and Flink checkpoints (v2_stream tables stay)
#
# Needs SENTINEL_DB_PASSWORD in the environment; it is passed to the
# containers, never written to a file.

param(
    [Parameter(Mandatory = $true)][ValidateSet("start", "submit", "replay", "status", "stop", "teardown")][string]$Command,
    [int]$Rate = 100,
    [int]$Customers = 4000
)

# "Continue", not "Stop": docker writes progress to stderr, which Windows
# PowerShell 5.1 would treat as a fatal error. Exit codes are checked instead.
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent (Split-Path -Parent $Here)
$Compose = @("compose", "-f", (Join-Path $Here "docker-compose.yml"))

if (-not $env:SENTINEL_DB_PASSWORD) { throw "Set SENTINEL_DB_PASSWORD first." }

switch ($Command) {
    "start" {
        & docker @Compose up -d --build
        if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
        & docker @Compose run --rm --no-deps replay-agent python simulator.py init
        if ($LASTEXITCODE -ne 0) { throw "init failed" }
    }
    "submit" {
        & docker @Compose exec -T jobmanager flink run -d -py /opt/sentinel/streaming/flink_job.py
        if ($LASTEXITCODE -ne 0) { throw "job submission failed" }
    }
    "replay" {
        $RunId = "stream_{0}eps_{1}" -f $Rate, (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss")
        & docker @Compose run --rm --no-deps replay-agent python simulator.py run --rate $Rate --customers $Customers --run-id $RunId
        if ($LASTEXITCODE -ne 0) { throw "replay failed" }
        & (Join-Path $Root "venv311\Scripts\python.exe") (Join-Path $Here "stream_check.py") --run-id $RunId
    }
    "status" {
        & docker @Compose ps
        try { (Invoke-RestMethod http://127.0.0.1:8081/jobs/overview).jobs | Format-Table name, state, start-time } catch { "Flink REST not reachable" }
    }
    "stop" { & docker @Compose stop }
    "teardown" { & docker @Compose down --volumes }
}
