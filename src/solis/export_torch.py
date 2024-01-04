import click
import torch


@click.command()
@click.argument("ckpt_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def export_torch(ckpt_path: str, output_path: str):
    checkpoint = torch.load(ckpt_path)
    state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        state_dict[key.removeprefix("model.")] = value
    torch.save({"state_dict": state_dict}, output_path)


if __name__ == "__main__":
    export_torch()
