from MSEModel import MSEModel

def create_model(model, input_dim):
    if model == 'siamese_gcntn': # TODO: fix
        return MSEModel(input_dim)
    else:
        raise RuntimeError('Unknown model {}'.format(model))