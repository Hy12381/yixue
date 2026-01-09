import argparse
import os

from delirium_web.modeling import train_delirium_model
from delirium_web.web import create_app


def train_web(*, data_path, external_path, output_dir):
    model_path = train_delirium_model(
        data_path=data_path,
        external_validation_path=external_path,
        output_dir=output_dir,
    )
    print(f"Web 模型已保存: {model_path}")


def serve_web(*, model_path, host, port):
    app = create_app(model_path=model_path)
    app.run(host=host, port=port, debug=False)


def _build_parser():
    parser = argparse.ArgumentParser(prog="delirium-web")
    sub = parser.add_subparsers(dest="cmd")

    train = sub.add_parser("train", help="训练7变量模型并保存")
    train.add_argument(
        "--data",
        default=r"e:\桌面图标\新建文件夹\相关图和表\data.csv",
    )
    train.add_argument(
        "--external",
        default=r"e:\桌面图标\新建文件夹\外部验证\外部验证集.csv",
    )
    train.add_argument(
        "--out",
        default=r"e:\桌面图标\新建文件夹\新建文件夹",
    )

    serve = sub.add_parser("serve", help="启动网页服务")
    serve.add_argument(
        "--model",
        default=os.path.join(
            r"e:\桌面图标\新建文件夹\新建文件夹",
            "delirium_web_model.joblib",
        ),
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5000)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.cmd:
        args.cmd = "serve"

    if args.cmd == "train":
        train_web(data_path=args.data, external_path=args.external, output_dir=args.out)
        return

    if args.cmd == "serve":
        serve_web(model_path=args.model, host=args.host, port=args.port)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")

