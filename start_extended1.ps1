param([int]$Port = 7861)

$projectDir = $PSScriptRoot
$pythonPath = Join-Path $projectDir '.venv\Scripts\python.exe'
$modelPath = Join-Path $projectDir 'models\extended1_bundle.json'
if (-not (Test-Path -LiteralPath $modelPath)) {
    $modelPath = Join-Path $projectDir 'models\extended1_selected.pth'
}
$appPath = Join-Path $projectDir 'ocr_review_app.py'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Chưa có môi trường .venv. Xem README.md để cài."
}
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "Chưa có model $modelPath. Chạy quy trình train trước."
}

Write-Host "Mở http://127.0.0.1:$Port sau khi Gradio khởi động. Giữ cửa sổ PowerShell này mở."
& $pythonPath $appPath --compare-weights $modelPath --default-model fine_tuned --port $Port
