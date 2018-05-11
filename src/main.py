#################### paramaters
model = 'iwge'
data = 'aids10k'
exp = 'try'
train = 1
####################

from utils import get_model_fun, load_data
model_func = get_model_fun(model, train)
data = load_data(data, train)

from time import time
if train:
    t = time()
    model_func(data, exp, exp == 'try')
    print('time={:.5f}'.format(time() - t))
else:
    print('TODO: sending one query at a time')



