from MSEModel import SiameseGCNTNMSE
from HingeModel import SiameseGCNTNHinge


def create_model(model, input_dim):
    if model == 'siamese_gcntn_mse':
        return SiameseGCNTNMSE(input_dim)
    elif model == 'siamese_gcntn_hinge':
        return SiameseGCNTNHinge(input_dim)
    else:
        raise RuntimeError('Unknown model {}'.format(model))
