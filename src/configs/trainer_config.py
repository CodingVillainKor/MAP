from dataclasses import dataclass

@dataclass
class TrainerConfig:
    trainer_name: str

    trainer_config: dict

    @classmethod
    def default(cls):
        return cls(
            trainer_name="default",
            trainer_config={
                "epochs": 10,
                "lr": 0.001
            }
        )