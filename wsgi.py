import os

from delirium_web.web import create_app


def _default_model_path():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "delirium_web_model.joblib"),
        os.path.join(here, "新建文件夹", "delirium_web_model.joblib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


MODEL_PATH = os.environ.get("MODEL_PATH", _default_model_path())

app = create_app(model_path=MODEL_PATH)
