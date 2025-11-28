# PowerShell script to run MLB win prediction commands
# Usage: .\run.ps1 [command]

$env:PYTHONPATH = "src"

$command = $args[0]

switch ($command) {
    "download" {
        Write-Host "Downloading MLB data..." -ForegroundColor Green
        python -m mlb_win_pred.data_download
    }
    "build" {
        Write-Host "Building dataset..." -ForegroundColor Green
        python -m mlb_win_pred.dataset_builder
    }
    "train" {
        Write-Host "Training models..." -ForegroundColor Green
        python -m mlb_win_pred.train
    }
    "evaluate" {
        Write-Host "Evaluating models..." -ForegroundColor Green
        python -m mlb_win_pred.evaluate
    }
    "predict" {
        if ($args.Length -lt 4) {
            Write-Host "Usage: .\run.ps1 predict --team TEAM --opponent OPP --date YYYY-MM-DD" -ForegroundColor Yellow
            exit 1
        }
        python -m mlb_win_pred.predict $args[1..($args.Length-1)]
    }
    "dash" {
        Write-Host "Starting Dash app..." -ForegroundColor Green
        python dashboard/dash_app.py
    }
    "streamlit" {
        Write-Host "Starting Streamlit app..." -ForegroundColor Green
        streamlit run dashboard/streamlit_app.py
    }
    default {
        Write-Host "MLB Win Prediction - Helper Script" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: .\run.ps1 [command]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Cyan
        Write-Host "  download   - Download MLB data"
        Write-Host "  build      - Build processed dataset"
        Write-Host "  train      - Train models"
        Write-Host "  evaluate   - Evaluate models"
        Write-Host "  predict    - Make a prediction (requires --team, --opponent, --game_date)"
        Write-Host "  dash       - Start Dash dashboard"
        Write-Host "  streamlit  - Start Streamlit dashboard"
        Write-Host ""
        Write-Host "Example workflow:" -ForegroundColor Cyan
        Write-Host "  .\run.ps1 download"
        Write-Host "  .\run.ps1 build"
        Write-Host "  .\run.ps1 train"
        Write-Host "  .\run.ps1 evaluate"
    }
}

