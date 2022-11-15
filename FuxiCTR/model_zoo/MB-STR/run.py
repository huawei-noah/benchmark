
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from src.model import RecModel
from src.datamodule import RecDataModule


def cli_main():
    cli = LightningCLI(RecModel, RecDataModule, save_config_overwrite=True)

if __name__ == '__main__':
    cli_main()