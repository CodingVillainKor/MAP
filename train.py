from src.trainer import Trainer

def main():
    trainer = Trainer()
    trainer.train(5)
    trainer.test()

if __name__ == "__main__":
    main()