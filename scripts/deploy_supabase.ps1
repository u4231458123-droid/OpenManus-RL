# Supabase Deployment Script

Write-Host "OpenManus-RL Supabase Deployment" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""

# Check if .env.supabase exists
if (-not (Test-Path ".env.supabase")) {
    Write-Host "ERROR: .env.supabase file not found!" -ForegroundColor Red
    Write-Host "Please create .env.supabase with your Supabase credentials." -ForegroundColor Yellow
    exit 1
}

# Check if Supabase CLI is installed
$supabaseCli = Get-Command supabase -ErrorAction SilentlyContinue
if (-not $supabaseCli) {
    Write-Host "Supabase CLI not found. Installing..." -ForegroundColor Yellow
    npm install -g supabase
}

# Load environment variables
Get-Content .env.supabase | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

$projectRef = $env:SUPABASE_PROJECT_REF
if (-not $projectRef) {
    Write-Host "ERROR: SUPABASE_PROJECT_REF not set in .env.supabase" -ForegroundColor Red
    exit 1
}

Write-Host "Linking to Supabase project: $projectRef" -ForegroundColor Cyan
supabase link --project-ref $projectRef

Write-Host ""
Write-Host "Pushing database migrations..." -ForegroundColor Cyan
supabase db push

Write-Host ""
Write-Host "Deploying Edge Functions..." -ForegroundColor Cyan
supabase functions deploy submit-rollout
supabase functions deploy log-agent-state
supabase functions deploy complete-rollout
supabase functions deploy get-metrics

Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install -q supabase python-dotenv

Write-Host ""
Write-Host "Uploading datasets to Supabase Storage..." -ForegroundColor Cyan
python scripts/upload_datasets.py

Write-Host ""
Write-Host "=================================" -ForegroundColor Green
Write-Host "Deployment Complete! ✓" -ForegroundColor Green
Write-Host ""
Write-Host "Your OpenManus-RL project is now live on Supabase!" -ForegroundColor Cyan
Write-Host "Project URL: https://$projectRef.supabase.co" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Set your database password in .env.supabase"
Write-Host "2. Run training with Supabase logging enabled"
Write-Host "3. View metrics at the Edge Function endpoint"
Write-Host ""
