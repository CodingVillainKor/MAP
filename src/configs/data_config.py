from dataclasses import dataclass

@dataclass
class DataConfig:
    dataset_name: str

    dataset_config: dict
    dataloader_config: dict

    @classmethod
    def mnist(cls):
        return cls(
            dataset_name="mnist",
            dataset_config={
                "mode": "train"
            },
            dataloader_config={
                "batch_size": 32,
                "shuffle": True
            }
        )