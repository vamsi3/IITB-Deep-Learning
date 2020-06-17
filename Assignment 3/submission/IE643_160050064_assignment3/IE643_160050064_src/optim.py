## AUTHOR: Vamsi Krishna Reddy Satti

##################################################################################
# Optimizers
##################################################################################


class SGD:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            for name in layer.params:
                layer.params[name] -= self.lr * layer.grad[name]
