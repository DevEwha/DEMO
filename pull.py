from huggingface_hub import snapshot_download
import os

# 어댑터 다운로드할 로컬 경로
adapter_path = "./models/25_pruning_AB_lora"
os.makedirs(adapter_path, exist_ok=True)

# 어댑터 파일 다운로드 (모든 파일 받아오기)
print("어댑터 다운로드 중...")
downloaded_path = snapshot_download(
    repo_id="dddreamerrr/25_pruning_AB_lora",
    local_dir=adapter_path,
    local_dir_use_symlinks=False  # 실제 파일로 저장
)

print(f"어댑터가 다운로드됨: {downloaded_path}")
print(f"다운로드된 파일 목록:")
for root, dirs, files in os.walk(adapter_path):
    for file in files:
        print(f"  {os.path.join(root, file)}")
