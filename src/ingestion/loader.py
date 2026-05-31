import os
import yaml
from pathlib import Path
from langchain_core.documents import Document


def load_dbt_files(dbt_path: str) -> list[Document]:
    """Walk a dbt project and return one Document per meaningful unit."""
    dbt_root = Path(dbt_path)
    models_dir = dbt_root / "models"
    documents = []

    for file_path in sorted(models_dir.rglob("*")):
        if file_path.suffix == ".sql":
            documents.extend(_load_sql_file(file_path, dbt_root))
        elif file_path.suffix in (".yml", ".yaml"):
            documents.extend(_load_yaml_file(file_path, dbt_root))

    return documents


def _load_sql_file(file_path: Path, dbt_root: Path) -> list[Document]:
    # One SQL file = one model = one chunk
    text = file_path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "file_path": str(file_path.relative_to(dbt_root)),
                "model_name": file_path.stem,
                "file_type": "sql",
            },
        )
    ]


def _load_yaml_file(file_path: Path, dbt_root: Path) -> list[Document]:
    # One YAML file can describe multiple models — one chunk per model entry
    text = file_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)

    # Not all YAML files have a 'models' key (e.g. dbt_project.yml)
    models = parsed.get("models", []) if parsed else []

    docs = []
    for model in models:
        model_name = model.get("name", "unknown")
        # Serialize the model entry back to YAML string so the text is readable
        chunk_text = yaml.dump(model, default_flow_style=False, sort_keys=False)
        docs.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "file_path": str(file_path.relative_to(dbt_root)),
                    "model_name": model_name,
                    "file_type": "yaml",
                },
            )
        )
    return docs
