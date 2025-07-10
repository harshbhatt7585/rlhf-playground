import os
import logging
from huggingface_hub import create_repo, upload_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_hf_token(explicit_token: str = "") -> str:
    """
    Return the HF token, preferring an explicit argument, then env var.
    """
    if explicit_token:
        return explicit_token
    env_token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    if env_token:
        return env_token
    raise EnvironmentError("No Hugging Face token provided. Set HUGGINGFACE_TOKEN or pass --token.")

def upload_to_hf(
    repo_id: str,
    folder_path: str = "./output",
    private: bool = False,
    exist_ok: bool = True,
    hf_token: str | None = None
) -> None:
    """
    Uploads the model folder to Hugging Face Hub.

    Args:
        repo_id (str): Hugging Face repo ID (e.g., "username/model-name")
        folder_path (str): Local path to model folder (e.g., "./output")
        private (bool): Whether to create a private repo
        exist_ok (bool): Skip creation if repo already exists
        hf_token (str): HF token (optional; will fallback to HUGGINGFACE_TOKEN)
    """
    token = get_hf_token(hf_token)
    logger.info(f"Token: {token}")

    logger.info(f"Creating HF repo: {repo_id} (private={private})")
    create_repo(repo_id=repo_id, private=private, exist_ok=exist_ok, token=token)

    logger.info(f"Uploading model from {folder_path} to {repo_id}")
    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        token=token,
        path_in_repo="",  # upload at repo root
    )

    logger.info(f"Model pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub.")
    parser.add_argument("repo_id", type=str,
                        help="Hugging Face repo ID (username/repo-name)")
    parser.add_argument("--folder", "-f", type=str, default="./output",
                        help="Folder containing model files")
    parser.add_argument("--private", "-p", action="store_true",
                        help="Create as private repository")
    parser.add_argument("--token", "-t", type=str, default="",
                        help="Hugging Face token (overrides HUGGINGFACE_TOKEN)")
    args = parser.parse_args()

    upload_to_hf(
        repo_id=args.repo_id,
        folder_path=args.folder,
        private=args.private,
        hf_token=args.token,
    )
