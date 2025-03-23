from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str

    model_config: dict

    @classmethod
    def net(cls):
        return cls(
            model_name="net",
            model_config={
                "dim": 64,
            }
        )