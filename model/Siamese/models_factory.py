from MSEModel import MSEModel
from HingeModel import HingeModel


def create_model(model, input_dim):
    if model == 'siamese_gcntn_mse':
        return MSEModel(input_dim)
    elif model == 'siamese_gcntn_hinge':
        return HingeModel(input_dim)
    else:
        raise RuntimeError('Unknown model {}'.format(model))
