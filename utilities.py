import itertools
import pandas as pd
import numpy as np
import inspect
import os
import concurrent.futures


def run_configuration(SimulationClass, conf, conf_number, seeds, verbose):
    results = []
    conf = dict(conf)
    run_number = 0
    for seed in seeds:
        sim = SimulationClass(seed=seed, **conf, configuration=conf_number, run=run_number)
        sim.go(verbose=False)
        results.append(sim)
        run_number += 1
    if verbose: print('.', end='')
    conf_number += 1
    if conf_number % 10 == 0:
        if verbose: print(conf_number)    
    return results

def run_configurations(SimulationClass, parameters_ranges, runs_per_configuration = 100, auto_seed=True, seeds = None, parallel=False, verbose = True):
    configurations = [[(k, v) for v in parameters_ranges[k]] for k in parameters_ranges]
    configurations = list(itertools.product(*configurations))

    if auto_seed:        
        seeds = range(runs_per_configuration)
    else:
        assert len(seeds) == run_configurations
    if verbose: print(f"number of configurations = {len(configurations)}")

    results = []
    if parallel:      
        cpu_count = os.cpu_count() - 1
        print(f'You have {cpu_count} CPUs that the simulation will use')       
        with concurrent.futures.ProcessPoolExecutor(cpu_count) as executor:
            futures = [executor.submit(run_configuration, SimulationClass,  conf, conf_number, seeds, verbose) for conf_number, conf in enumerate(configurations)]
        for f in concurrent.futures.as_completed(futures):
            results.extend(f.result())
        
    else:
        for conf_number, conf in enumerate(configurations):
            result = run_configuration(SimulationClass, conf, conf_number, seeds, verbose)
            results.extend(result)

    if verbose: 
        print('Done running the simulations!')
        print('Assembling the results ...')

    return results

def extract_history(simulation):
    attributes = inspect.getmembers(simulation, lambda a:not(inspect.isroutine(a)))
    public_attributes = [(a, v) for a, v in attributes if not a.startswith('_')]
    d = {}
    for a, v in public_attributes:
        a = a.replace('_', ' ').title() # format
        if a.endswith(' S'):
            a = a[:-2] # remove trailing s for singular noun
        if type(v) == type([]):
            d[a] = v # history attributes
        else:
            d[a] = (simulation.simulation_time + 1)* [v] # single value attributes, repeat per each step
    data = pd.DataFrame(d)
    return data

def round_float_values_in_data_frame(a_data_frame):
    for col, dtype in a_data_frame.dtypes.to_dict().items():
        if str(dtype) == 'float64':
            a_data_frame[col] = np.round(a_data_frame[col], decimals=5)    
    return a_data_frame

def create_result_table(list_of_simulations, history=True, round_values=True, agg_function='last'):
    data = pd.concat((extract_history(s) for s in list_of_simulations))
    data = data.sort_values(['Configuration', 'Run', 'Time Step'])
    if not history:
        data = data.groupby(['Configuration', 'Run']).aggregate(agg_function).sort_index().reset_index()
    if round_values: data = round_float_values_in_data_frame(data)
    return data

# test code
if __name__ == '__main__':
    from esm_simulation import * # needed for testing
    params = {'communication_transparency': [True, False],}
    SimulationClass = Simulation
    results = run_configurations(SimulationClass, params, parallel=False, runs_per_configuration=5)
    data = create_result_table(results)
    sim = results[0]
    print(data.head())
    print()