from dataclasses import dataclass

from .data_config import DataConfig
from .model_config import ModelConfig
from .trainer_config import TrainerConfig

@dataclass
class Config:
    exp_name: str

    data_config: DataConfig
    model_config: ModelConfig
    trainer_config: TrainerConfig

    @classmethod
    def get_config(cls, exp_name):
        if exp_name == "exp1":
            return cls(
                exp_name=exp_name,
                data_config=DataConfig.mnist(),
                model_config=ModelConfig.net(),
                trainer_config=TrainerConfig.default()
            )