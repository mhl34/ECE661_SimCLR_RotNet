import math

# Hyperparameters for SimCLR
class hyperParams:
    def __init__(self):
        self.batch_size = 128
        self.projection_dim = 128
        self.temperature = 0.5
        self.learning_rate = 0.075 * math.sqrt(self.batch_size)
        # self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.epochs = 100
        self.num_classes = 10
        self.labeling_fraction = 0.1