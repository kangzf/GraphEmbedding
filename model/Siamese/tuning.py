# AUTO TUNING PARAMETERS
# Hyperparameters   | Range        | Note
#-----------------------------------------------------------------------------------------------------------------------
# norm_dist          T/F            2 norm/no norm
# yeta               0.2/0.01       5 diff yetas
# activation func    5              5 diff  activation layers
# learning_rate      0.06-0.001     6 diff lr
# iteration          500-2000       4 diff iter
# validation ratio   0.2-0.4        2 diff ratio   
#-----------------------------------------------------------------------------------------------------------------------
import csv
import itertools
from train_tuning import train, reset_graph
from os import system

file_name = 'parameter_tuning_aids50.csv'

header = ['norm_dist','yeta','final_act','learning_rate','iter','validation_ratio',
          'best_train_loss','best_train_loss_iter','best_val_loss','best_val_loss_iter',
          'prec@1_norm', 'prec@2_norm', 'prec@3_norm', 'prec@4_norm', 'prec@5_norm','prec@6_norm','prec@7_norm','prec@8_norm','prec@9_norm',
          'prec@10_norm','prec@1_nonorm', 'prec@2_nonorm', 'prec@3_nonorm', 'prec@4_nonorm', 'prec@5_nonorm','prec@6_nonorm','prec@7_nonorm',
          'prec@8_nonorm','prec@9_nonorm','prec@10_nonorm','mrr_norm','mrr_nonorm','mse_norm','mse_nonorm']

norm_dist_range = [True, False]
yeta_range_norm = [0.1,0.2,0.3,0.5,0.8]
yeta_range_nonorm = [0.1,0.05,0.01,0.005,0.001]
final_act_range = ['identity', 'relu', 'sigmoid', 'tanh', 'sim_kernel']
lr_range = [0.06,0.03,0.01,0.006,0.003,0.001]
iter_range = list(range(500,2001,500))
val_ratio_range = [0.2,0.4]

best_result_train_loss, best_result_val_loss = float('Inf'), float('Inf')
best_result_train_iter, best_result_val_iter = 0, 0

result_train = [] 
result_val = []

i = 1

File = open(file_name, 'w')
with File:
    writer = csv.writer(File)
    writer.writerows([header])
    
def parse_result(result):
    mrr_norm = result['mrr_norm']['siamese_gcntn']
    mrr_nonorm = result['mrr_nonorm']['siamese_gcntn']
    mse_norm = result['mse_norm']['siamese_gcntn']
    mse_nonorm = result['mse_nonorm']['siamese_gcntn']
    apk_norm = result['apk_norm']['siamese_gcntn']['aps'][:10]
    apk_nonorm = result['apk_nonorm']['siamese_gcntn']['aps'][:10]
    model_result = list(apk_norm)+list(apk_nonorm)+[mrr_norm,mrr_nonorm,mse_norm,mse_nonorm]
    return model_result

def csv_record(mesg):
    global file_name
    File = open(file_name, 'a')
    with File:
        writer = csv.writer(File)
        writer.writerows(mesg)

for norm_dist in norm_dist_range:
    yeta_range = yeta_range_norm if norm_dist==True else yeta_range_nonorm
    for yeta, final_act, lr, iteration, val_ratio in itertools.product(yeta_range, final_act_range, lr_range, iter_range, val_ratio_range):
        print('Number of tuning iteration: {}'.format(i), 'norm_dist: {}'.format(norm_dist), 'yeta: {}'.format(yeta), 'final_act: {}'.format(final_act),
              'learning_rate: {}'.format(lr), 'iteration: {}'.format(iteration), 'validation_ratio: {}'.format(val_ratio))
        i += 1
        try:
            best_train_loss, best_train_iter, best_val_loss, best_val_iter, result = train(valid_percentage=val_ratio, norm_dist=norm_dist, 
                                                                                yeta=yeta, final_act=final_act,learning_rate=lr, iters=iteration)
            print('best_train_loss: {}'.format(best_train_loss), 'best_val_loss: {}'.format(best_val_loss),
                  'best_train_iter: {}'.format(best_train_iter), 'best_val_iter: {}'.format(best_val_iter))
        except:
            csv_record([['Exception:']])
            csv_record([[str(x) for x in [norm_dist,yeta,final_act,lr,iteration,val_ratio]]])
            reset_graph()
            continue
        else:
            model_result = parse_result(result)
            csv_record([[str(x) for x in [norm_dist,yeta,final_act,lr,iteration,val_ratio,
                    best_train_loss,best_train_iter,best_val_loss,best_val_iter]+model_result]])

            if best_train_loss < best_result_train_loss:
                best_result_train_loss = best_train_loss
                result_train = [norm_dist,yeta,final_act,lr,iteration,val_ratio,
                              best_train_loss,best_train_iter]+model_result

            if best_val_loss < best_result_val_loss:
                best_result_val_loss = best_val_loss
                result_val = [norm_dist,yeta,final_act,lr,iteration,val_ratio,
                              best_val_loss,best_val_iter]+model_result
            
print(result_train)
print(result_val)

csv_record([['Final Results:']])
csv_record([['norm_dist','yeta','final_act','learning_rate','iter','validation_ratio',
          'best_train_loss','best_train_loss_iter',
          'prec@1_norm', 'prec@2_norm', 'prec@3_norm', 'prec@4_norm', 'prec@5_norm','prec@6_norm','prec@7_norm','prec@8_norm','prec@9_norm',
          'prec@10_norm','prec@1_nonorm', 'prec@2_nonorm', 'prec@3_nonorm', 'prec@4_nonorm', 'prec@5_nonorm','prec@6_nonorm','prec@7_nonorm',
          'prec@8_nonorm','prec@9_nonorm','prec@10_nonorm','mrr_norm','mrr_nonorm','mse_norm','mse_nonorm']])
csv_record([[str(x) for x in result_train]])

csv_record([['norm_dist','yeta','final_act','learning_rate','iter','validation_ratio',
          'best_val_loss','best_val_loss_iter',
          'prec@1_norm', 'prec@2_norm', 'prec@3_norm', 'prec@4_norm', 'prec@5_norm','prec@6_norm','prec@7_norm','prec@8_norm','prec@9_norm',
          'prec@10_norm','prec@1_nonorm', 'prec@2_nonorm', 'prec@3_nonorm', 'prec@4_nonorm', 'prec@5_nonorm','prec@6_nonorm','prec@7_nonorm',
          'prec@8_nonorm','prec@9_nonorm','prec@10_nonorm','mrr_norm','mrr_nonorm','mse_norm','mse_nonorm']])
csv_record([[str(x) for x in result_val]])


