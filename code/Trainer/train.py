from Trainer import Trainer

for layer in ["L3m"]:
    trainer_x = Trainer("config.json",layer,"x")
    trainer_x.train()
    trainer_x.test()
    trainer_x.visualize()

    # trainer_y = Trainer("config.json",layer,"y")
    # trainer_y.train()
    # trainer_y.test()
    # trainer_y.visualize()
