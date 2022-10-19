from collections import defaultdict
import numpy as np
import pulp as plp
# import docplex.mp.model as cpx
import pandas as pd
import time

#### data acquisition ####

# defiable a class for item
class Item:
    def __init__(self, width, height, demand, number):
        self.width = width
        self.height = height
        self.demand = demand
        self.number = number

    def __eq__(self, other):
        return self.width == other.width and self.height == other.height 

    def __hash__(self):
        return hash((self.width, self.height))

    def __repr__(self):
        return f'Item({self.width}, {self.height}, {self.demand})'

# read data from excel
print('reading data...')
data_path = '/mnt/nfs/zhshen/works/jianmo/dataA1.csv'
data = pd.read_csv(data_path)
data['area'] = data['item_length'] * data['item_width']
data = data.sort_values(by=["area"], ascending=False)
data.reset_index(drop=True, inplace=True)
lengths = data['item_length']
widths = data['item_width']
bigtmp = []
tmp = [(lengths[0], widths[0])]
bigtmp.append(tmp)
# print(bigtmp)
for i in range(1, len(lengths)):
    flag = False
    for item in bigtmp:
        # print(item)
        if np.abs(lengths[i] - item[0][0]) < 100 and np.abs(widths[i] - item[0][1]) < 100:
            item.append((lengths[i], widths[i]))
            flag = True
            break
    if not flag:
        bigtmp.append([(lengths[i], widths[i])])
print('total types:',len(bigtmp))
items = []
#find each type item max length and max width
# item_types = []
for i,tmps in enumerate(bigtmp):
    max_length = 0
    max_width = 0
    for tmp in tmps:
        if tmp[0] > max_length:
            max_length = tmp[0]
        if tmp[1] > max_width:
            max_width = tmp[1]
    items.append(Item(max_length, max_width, len(tmps),i))
    # if i>80:
    #     cnt = 0
    #     for j in range(i,len(bigtmp)):
    #         cnt += len(bigtmp[j])
    #     items.append(Item(max_length, max_width, cnt,i))
    #     break


# for i in range(len(data)):
#     item = Item(float(data['item_length'][i]), float(data['item_width'][i]), int(data['item_num'][i]), i)
#     # print(item)
#     items.append(item)
#     if i > 100:
#         break

# item_1 = Item(4, 3, 5,0)
# item_2 = Item(2, 2, 5,1)

# items = [item_1, item_2]



#### model construction ####
class Plate:
    def __init__(self, width, height, stage):
        self.width = width
        self.height = height
        self.stage = stage
    
    def __eq__(self, other):
        return self.width == other.width and self.height == other.height

    def __hash__(self):
        return hash((self.width, self.height))

    def __repr__(self):
        return f'Plate({self.width}, {self.height}, {self.stage})'


class Cut:
    def __init__(self, plate, item):
        self.plate = plate
        self.item = item
        self.analysed = False

    def __eq__(self, other):
        return self.plate == other.plate and self.item == other.item

    def __hash__(self):
        return hash((self.plate, self.item))
    
    def __repr__(self):
        return f'Cut({self.plate}, {self.item})'

def table_2(plate,item,minh,minw):
    if plate.stage == 1:
        top_plate = Plate(plate.width, plate.height - item.height, 1)
        right_plate = Plate(plate.width - item.width, item.height, 2)
    elif plate.stage == 2:
        top_plate = Plate(item.width, plate.height - item.height, 3)
        right_plate = Plate(plate.width - item.width, plate.height, 2)
    else:
        top_plate = None
        right_plate = None
    if top_plate is not None and top_plate.height < minh:
        top_plate = None
    if right_plate is not None and right_plate.width < minw:
        right_plate = None
    return top_plate, right_plate

def table_3(plate, item, minh, minw):
    if plate.stage == 1:
        top_plate = Plate(plate.width, plate.height - item.height, 1)
        right_plate = Plate(plate.width - item.width, item.height, 2)
    elif plate.stage == 2:
        top_plate = Plate(item.width, plate.height - item.height, 3)
        right_plate = Plate(plate.width - item.width, plate.height, 2)
    elif plate.stage == 3:
        top_plate = Plate(plate.width, plate.height - item.height, 3)
        right_plate = Plate(plate.width - item.width, item.height, 4)
    else:
        top_plate = None
        right_plate = None
    if top_plate is not None and top_plate.height < minh:
        top_plate = None
    if right_plate is not None and right_plate.width < minw:
        right_plate = None
    return top_plate, right_plate

initial_plate = Plate(2440, 1220, 1)
# initial_plate = Plate(6, 6, 1)
R = defaultdict(int)
C = []
min_h = min([item.height for item in items])
min_w = min([item.width for item in items])
for item in items:
    C.append(Cut(initial_plate, item))

R[initial_plate]=0
result_relationship = []
tmp_number = 0
C_collec = set()
print('enumerating cuts...')
while not len(C)==0:
    cut = C.pop(0)
    C_collec.add(cut)
    # cut.analysed = True
    top_plate, right_plate = table_2(cut.plate, cut.item, min_h, min_w)
    if top_plate is not None:
        if R[top_plate]==0:
        # if top_plate not in R.keys():
            R[top_plate] = tmp_number+1
            tmp_number += 1
        result_relationship.append((cut.item.number, R[cut.plate], R[top_plate]))
        
    if right_plate is not None:
        if R[right_plate]==0:
        # if right_plate not in R.keys():
            R[right_plate] = tmp_number+1
            tmp_number += 1
        result_relationship.append((cut.item.number, R[cut.plate],  R[right_plate]))
        
    for item in items:
        if top_plate is not None:
            # if top_plate.stage == 3:
            #     if item.width == top_plate.width and item.height <= top_plate.height:
            #         C.append(Cut(top_plate, item))
            # else:
            if item.width <= top_plate.width and item.height <= top_plate.height:
                C.append(Cut(top_plate, item))
        if right_plate is not None:
            # if right_plate.stage == 2:
            #     if item.width <= right_plate.width and item.height == right_plate.height:
            #         C.append(Cut(right_plate, item))
            # else:
            if item.width <= right_plate.width and item.height == right_plate.height:
                C.append(Cut(right_plate, item))

    if len(R)>=2000:
        break

# print(R)
# print(C)


C_list = [(cut.item.number, R[cut.plate]) for cut in C_collec]
# print(C_list)
num_items = len(items)
num_plates = len(R)
# for r in R:
#     print(r.width, r.height, r.stage, R[r])
print('len items',num_items)
print('len plates',num_plates)
print('len cuts',len(C_list))
a = np.zeros((num_items, num_plates, num_plates))
# print(a.shape)
for relation in result_relationship:
    a[relation[0],relation[1],relation[2]] = 1
# print(result_relationship)
# print(a.sum())

# raise Exception('stop here')


#### model solving ####
print('solving model...')
opt_model = plp.LpProblem(name="MIP_Model")
x_vars  = {(i,j):plp.LpVariable(cat=plp.LpInteger, 
               lowBound=0, upBound= 1000,
               name="x_{0}_{1}".format(i,j)) for i in range(num_items) for j in range(0,num_plates)}

constraints = {i : opt_model.addConstraint(
plp.LpConstraint(
             e=plp.lpSum(x_vars[i,j] for j in range(0,num_plates)),
             sense=plp.LpConstraintGE,
             rhs=items[i].demand,
             name="constraint_{0}".format(i)))for i in range(num_items)}

constraints_a = {k : opt_model.addConstraint(
plp.LpConstraint(
             e=plp.lpSum(a[i][j][k] * x_vars[i,j] for i in range(num_items) for j in range(0,num_plates))-plp.lpSum(x_vars[i,k] for i in range(num_items)),
             sense=plp.LpConstraintGE,
             rhs=0,
             name="constraint_a_{0}".format(k)))for k in range(1,num_plates)}

constraints_b = {(i,j): opt_model.addConstraint(
plp.LpConstraint(e=x_vars[i,j],
        sense=plp.LpConstraintEQ,
        rhs=0,
        name="constraint_b_{0}_{1}".format(i,j))) 
        for i in range(num_items) for j in range(1,num_plates) if (i,j) not in C_list}

objective = plp.lpSum(x_vars[i,0] for i in range(num_items))
opt_model.sense = plp.LpMinimize
opt_model.setObjective(objective)
slover = plp.PULP_CBC_CMD(timeLimit=7200,msg=0)
opt_model.solve(slover)
# opt_model.solve(solver = GLPK_CMD())
# print('status:', plp.LpStatus[opt_model.status])
print('objective:', plp.value(opt_model.objective))

# opt_model = cpx.Model(name="MIP Model")
# x_vars  = {(i,j): opt_model.integer_var(lb=0, ub= 999,name="x_{0}_{1}".format(i,j))
#      for i in range(num_items) for j in range(0,num_plates)}


# constraints = {i : opt_model.add_constraint(
#     ct=opt_model.sum(x_vars[i,j] for j in range(0,num_plates))>=items[i-1].demand,
#     ctname="constraint_{0}".format(i))
#         for i in range(num_items)}


# constraints_a = {k : opt_model.add_constraint(
#     ct=opt_model.sum(a[i][j][k] * x_vars[i,j] for i in range(num_items) for j in range(0,num_plates))
#     >= opt_model.sum(x_vars[i,k] for i in range(num_items)),
#     ctname="constraint_a_{0}".format(k))
#         for k in range(1,num_plates)}

# # constraints_b = {opt_model.add_constraint(
# #     ct=x_vars[i,j] == 0,
# #     ctname="constraint_b_{0}_{1}".format(i,j))
# #         for i in range(num_items) for j in range(0,num_plates) if (i,j) not in C_list}

# objective  = opt_model.sum(x_vars[i,0] for i in range(num_items))

# opt_model.minimize(objective)

# st_time = time.time()
# opt_model.solve()
# end_time = time.time()
# print('solving time: ', end_time - st_time)

# # print result values
# print("objective value: {0}".format(opt_model.objective_value))
# print("Solution: ")
# for i in range(num_items):
#     for j in range(0,num_plates):
#         print("x_{0}_{1} = {2}".format(i,j,x_vars[i,j].solution_value))

# from ortools.linear_solver import pywraplp
# solver = pywraplp.Solver.CreateSolver('SCIP')
# ### variables ###
# # Assign variable:
# assign={}
# for r in range(0,num_items):
#     for j in range(0,num_plates):
#         assign[r,j] = solver.IntVar(0, 1000, 'Assign[%r][%d]' % (r, j))


# # objective: minimize number of techs:
# z = solver.Sum((assign[r,0]) for r in range(num_items))

# ### constraints ###
# # Assign:
# for r in range(num_items):
#     solver.Add(solver.Sum(assign[r,j] for j in range(0,num_plates))>=items[r-1].demand)

# # Bin capacity:
# for k in range(1,num_plates):
#     solver.Add(solver.Sum(a[r][j][k] * assign[r,j] for r in range(num_items) for j in range(0,num_plates))
#                 >= solver.Sum(assign[r,k] for r in range(num_items)))

# ## Objective:
# solver.Minimize(z)
# #
# # solution and search
# #
# solver.Solve()

# print()
# print('z: ', int(solver.Objective().Value()))
# # print('Assign:')
# # for r in range(num_items):
# #     for j in range(0,num_plates):
# #         print('Assign[%r][%d] = %d' % (r, j, assign[r,j].solution_value()))

# print()
# print('walltime  :', solver.WallTime(), 'ms')





