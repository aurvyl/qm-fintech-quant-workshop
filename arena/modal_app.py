"""Deploy the arena on Modal. Steps are in arena/README.md.

The arena reaches the workshop model through the workshop model server, with its own access code, so
the Anthropic key, the model and the limits stay on that server and never appear in this folder.
Settings come from the Modal secret "workshop-arena".
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas==3.0.6", "numpy==2.5.3", "matplotlib==3.11.2", "fastapi==0.162.0", "uvicorn==0.54.0")
    .add_local_python_source("backtester", "arena")
    .add_local_file(ROOT / "data" / "prices.csv", "/root/data/prices.csv")
)
app = modal.App("workshop-arena", image=image)
saved = modal.Dict.from_name("workshop-arena-state", create_if_missing=True)


class DictPersistence:
    def load(self):
        return saved.get("state", None)

    def save(self, state: dict) -> None:
        saved.put("state", state)


@app.function(secrets=[modal.Secret.from_name("workshop-arena")], min_containers=1, max_containers=1, timeout=3600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="workshop-arena")
def web():
    from arena.server import create_app, settings_from_env
    from backtester.agent import workshop_model

    ask = workshop_model(os.environ["ARENA_PROXY_URL"], os.environ["ARENA_PROXY_CODE"], seconds_between_calls=0.0)
    return create_app(settings_from_env(ask, DictPersistence()))
