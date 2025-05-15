'''
Eval script to produce a table of metrics.
@input: python tabulate.py <path_to_model1> <path_to_model2> ...
'''

from tabulate import tabulate
import json 
import glob 
import os 
import sys

exps = sys.argv[1:-1]
outname = sys.argv[-1]
data = []
for exp in exps:
    exp_name = exp.split('/')[-1]
    exp_data = [exp_name]
    # read metrics
    exp_path = os.path.dirname(os.path.dirname(exp))
    json_path = os.path.join(exp_path, f'results{outname}.json')
    with open(json_path) as file:
        metrics = json.load(file)
    exp_data.append(metrics['over_all'])
    exp_data.append(metrics['over_all_normal'])
    data.append(exp_data)
    
print(tabulate(data, headers=[f'Experiment Name at {outname}','Chamfer Distance (Mean)','Chamfer Distance (Plus Normal)'], tablefmt='github'))

data = []
for exp in exps:
    exp_name = exp.split('/')[-1]
    exp_data = [exp_name]
    # read metrics
    exp_path = os.path.dirname(os.path.dirname(exp))
    json_path = os.path.join(exp_path, f'results{outname}_lcc.json')
    with open(json_path) as file:
        metrics = json.load(file)
    exp_data.append(metrics['over_all'])
    exp_data.append(metrics['over_all_normal'])
    data.append(exp_data)
    
print(tabulate(data, headers=[f'Experiment Name at {outname}','Chamfer Distance (Mean) LCC','Chamfer Distance (Plus Normal) LCC'], tablefmt='github'))
    
    
