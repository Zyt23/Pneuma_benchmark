$ErrorActionPreference = "Stop"

python examples/qar_context60/make_representative_data.py --clean --output examples/qar_context60/data
python examples/qar_context60/run_evaluation_examples.py --clean --data-root examples/qar_context60/data --output examples/qar_context60/outputs/smoke

Write-Host "Smoke outputs: examples/qar_context60/outputs/smoke"
