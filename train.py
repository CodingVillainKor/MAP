import argparse

from src.configs.config import Config
from src.dataset import get_dataloader
from src.models import get_model
from src.trainer import Trainer

def main(args):
    config = Config.get_config(args.exp_name)
    dataloader = get_dataloader(config.data_config)
    model = get_model(config.model_config)
    trainer = Trainer(model, dataloader, config.trainer_config)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    main(args)