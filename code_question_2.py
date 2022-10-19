
from collections import defaultdict
import pulp as plp
import pandas as pd
import numpy as np
import time

file = '/mnt/nfs/zhshen/works/jianmo/dataB3.csv'
batch_num = 60





data = pd.read_csv(file)
order_materials = defaultdict(list)
for i in range(len(data)):
    order_materials[data.iloc[i]['item_order']].append(data.iloc[i]['item_material'])
def compare_materials(x, y):
    same_cnt = 0
    min_list = x if len(x) < len(y) else y
    max_list = x if len(x) > len(y) else y
    for i in min_list:
        if i in max_list:
            same_cnt += 1
    return 1 - same_cnt / len(min_list)



data['area'] = data['item_width'] * data['item_length']
order_data = data.groupby('item_order')[['item_num','area']].sum()
order_data = order_data.reset_index()

num_orders = order_data.shape[0]
order_itemCnts = order_data['item_num'].values
order_areas = order_data['area'].values
distance = np.zeros((num_orders, num_orders))
## distance matrix initialize
for i in range(num_orders):
    distance[i][i] = 0
    for j in range(i+1,num_orders):
        ratio= compare_materials(order_data.iloc[i]['item_order'], order_data.iloc[j]['item_order'])
        if ratio<0 or ratio>1:
            raise ValueError('ratio should be in [0,1]')
        distance[i][j] = ratio
        distance[j][i] = ratio

print('distance:',distance.sum())

#### model solving ####
print('solving model...')
opt_model = plp.LpProblem(name="MIP_Model")
x_vars  = {(i,j):plp.LpVariable(cat=plp.LpBinary, 
               name="x_{0}_{1}".format(i,j)) for i in range(num_orders) for j in range(0,num_orders)}

constraints_0 = {i : opt_model.addConstraint(
plp.LpConstraint(
             e=plp.lpSum(x_vars[i,j] for j in range(num_orders)),
             sense=plp.LpConstraintEQ,
             rhs=1,
             name="constraint_0_{0}".format(i)))for i in range(num_orders)}

constraints_1 = {opt_model.addConstraint(
plp.LpConstraint(
             e=plp.lpSum(x_vars[j,j] for j in range(num_orders)),
             sense=plp.LpConstraintEQ,
             rhs=batch_num,
             name="constraint_1_{0}".format(0)))}

constraints_2 = {(i,j) : opt_model.addConstraint(
plp.LpConstraint(
                e=x_vars[i,j]-x_vars[j,j],
                sense=plp.LpConstraintLE,
                rhs=0,
                name="constraint_2_{0}_{1}".format(i,j))) for i in range(num_orders) for j in range(num_orders)}

constraints_3 = {(j) : opt_model.addConstraint(
plp.LpConstraint(
                e=plp.lpSum(x_vars[i,j]*order_itemCnts[i] for i in range(num_orders)),
                sense=plp.LpConstraintLE,
                rhs=1000,
                name="constraint_3_{0}".format(j))) for j in range(num_orders)}

constraints_4 = {(j) : opt_model.addConstraint(
plp.LpConstraint(
                e=plp.lpSum(x_vars[i,j]*order_areas[i] for i in range(num_orders)),
                sense=plp.LpConstraintLE,
                rhs=250*1000000,
                name="constraint_4_{0}".format(j))) for j in range(num_orders)}



objective = plp.lpSum(x_vars[i,j]*distance[i][j] for i in range(num_orders) for j in range(num_orders))
opt_model.sense = plp.LpMinimize
opt_model.setObjective(objective)
slover = plp.PULP_CBC_CMD(timeLimit=120,msg=0)
st = time.time()
opt_model.solve(slover)
print('solving time: ', time.time()-st)
# opt_model.solve(solver = GLPK_CMD())
# print('status:', plp.LpStatus[opt_model.status])
print('objective:', plp.value(opt_model.objective))
# get solution
x_mat = np.zeros((num_orders, num_orders))
for i in range(num_orders):
    for j in range(num_orders):
        x_mat[i][j] = plp.value(x_vars[i,j])
np.save(file.replace('.csv','.npy'), x_mat)
# x_mat = np.load(file.replace('.csv','.npy'))

# print(x_mat.sum(axis=0))

# get batch
batch = defaultdict(int)
col_batch_name = defaultdict(int)
cnt = 0
for i in range(num_orders):
    if x_mat[i][i]>0:
        col_batch_name[i] = cnt
        cnt += 1
for i in range(num_orders):
    for j in range(num_orders):
        if x_mat[i][j] >0:
            batch[order_data.iloc[i]['item_order']] = col_batch_name[j]
            

data['batch'] = -1
for i in range(len(data)):
    data.loc[i,'batch'] = batch[data.loc[i,'item_order']]

data.to_csv(file.replace('.csv','_batch.csv'), index=False)