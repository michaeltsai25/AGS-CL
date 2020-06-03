import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, trainer, taskcla):
        
            
        if dataset == 'CUB200':
            if trainer == 'hat':
                import networks.alexnet_hat as alex
                return alex.alexnet(taskcla, pretrained=True)
            else:
                import networks.alexnet as alex
                return alex.alexnet(taskcla, pretrained=True)
