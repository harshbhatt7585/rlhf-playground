import os
import logging
from huggingface_hub import create_repo, upload_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_hf_token() -> str:
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise EnvironmentError("HUGGINGFACE_TOKEN environment variable not set.")
    return token

def upload_to_hf(
    repo_id: str,
    folder_path: str = "./output",
    private: bool = False,
    exist_ok: bool = True,
):
    """
    Uploads the model folder to Hugging Face Hub.

    Args:
        repo_id (str): Hugging Face repo ID (e.g., "username/model-name")
        folder_path (str): Local path to model folder (e.g., "./output")
        private (bool): Whether to create a private repo
        exist_ok (bool): Skip creation if repo already exists
    """
    token = get_hf_token()

    logger.info(f"Creating HF repo: {repo_id} (private={private})")
    create_repo(repo_id=repo_id, private=private, exist_ok=exist_ok, token=token)

    logger.info(f"Uploading model from {folder_path} to {repo_id}")
    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        token=token,
        path_in_repo="",  # upload at root of repo
    )

    logger.info(f"Model pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub.")
    parser.add_argument("repo_id", type=str, help="Hugging Face repo ID (username/repo-name)")
    parser.add_argument("--folder", type=str, default="./output", help="Folder containing model files")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    args = parser.parse_args()

    upload_to_hf(
        repo_id=args.repo_id,
        folder_path=args.folder,
        private=args.private,
    )
