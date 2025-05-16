import sys
from pathlib import Path
from huggingface_hub import snapshot_download

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

models_dir = project_root / "models" / "local" / "Qwen3-14B"
models_dir.mkdir(parents=True, exist_ok=True)

model_id = "Qwen/Qwen3-14B"
output_dir = models_dir

print(f"Project root: {project_root}")
print(f"Model will be downloaded to: {output_dir}")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False
    )
    print(f"Model files downloaded successfully to {output_dir}")
except Exception as e:
    print(f"Failed to download model: {str(e)}")
