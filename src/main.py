#################### paramaters
model = 'iwge'
data = 'syn'
exp = 'try'
train = 1
####################

from utils import get_model_fun, get_data
model_func = get_model_fun(model, train)
data = get_data(data, train)

from time import time
if train:
    t = time()
    model_func(data, exp, exp == 'try')
    print('time={:.5f}'.format(time() - t))
else:
    print('TODO: sending one query at a time')



