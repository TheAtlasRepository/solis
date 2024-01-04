from importlib import import_module

import click


@click.command()
@click.argument("model_path")
@click.argument("ckpt_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def export_onnx(model_path: str, ckpt_path: str, output_path: str):
    module_path = ".".join(model_path.split(".")[:-1])
    model_name = model_path.split(".")[-1]
    model_type = getattr(import_module(module_path), model_name)
    model = model_type.load_from_checkpoint(ckpt_path)
    model.to_onnx(output_path, export_params=True)


if __name__ == "__main__":
    export_onnx()
