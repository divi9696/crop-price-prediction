# Test API connectivity
Write-Host "Testing Crop Price Prediction API..." -ForegroundColor Green

# Test health endpoint
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method GET
    Write-Host "✅ Health check passed:" -ForegroundColor Green
    $health | ConvertTo-Json
}
catch {
    Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test prediction endpoint
Write-Host "`nTesting prediction endpoint..." -ForegroundColor Yellow
$predictionBody = @{
    market = "Kolkata"
    state = "West Bengal" 
    crop = "rice"
    variety = "PR 126"
    horizon_weeks = 4
    as_of_date = "2024-10-01"
} | ConvertTo-Json

try {
    $prediction = Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method Post -Body $predictionBody -ContentType "application/json"
    Write-Host "✅ Prediction successful:" -ForegroundColor Green
    $prediction | ConvertTo-Json -Depth 3
}
catch {
    Write-Host "❌ Prediction failed: $($_.Exception.Message)" -ForegroundColor Red
}