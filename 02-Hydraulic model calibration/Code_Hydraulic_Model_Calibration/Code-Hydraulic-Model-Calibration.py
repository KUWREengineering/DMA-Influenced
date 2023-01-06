import wntr
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
#%% input data
######################### Network ######################################
inp_initial                 = './Trunkmian_Before_Calibration.inp'
inp_after_calibration       = './Trunkmian_After_Calibration.inp'
##################### .DAT&.XLSX file ##################################
flow_res_obs                = './3res_FlowOBS.dat'
inlet_press_obs             = './Sources(Junction)_Pressure_Observe_data.dat'
p_obs                       = './junction_pressure_observe.dat'
f_obs                       = './link_flow_observe_for_calibrate.dat'
pipe                        = './pipe_information.xlsx'
########################### User #######################################
C_age                       = 3.5/30
year_calibrate              = 2565
max_iteration               = 5
#%% Setting defualts of hydraulic model
wn                                  = wntr.network.WaterNetworkModel(inp_initial)
wn.options.time.duration            = 23*3600
wn.options.time.hydraulic_timestep  = 3600
wn.options.time.report_timestep     = 3600
#%% Getting attribute of pipe
## Read text file 
inp_file                    = open(inp_initial ,'r')                                
inp_data                    = inp_file.readlines()
inp_file.close() 
## find index to select pipe information only 
pipes_index                 = inp_data.index("[PIPES]\n") 
pumps_index                 = inp_data.index("[PUMPS]\n")
inp_data                    = inp_data[pipes_index+1:pumps_index-1]
## rearrange to dataframe
pipe_columns                = pd.Series(inp_data[0].split('\t')).str.replace(';', "")
pipe_columns                = list(pipe_columns.str.strip())+['Description']
pipe_information            = pd.DataFrame(index=pipe_columns,columns=np.arange(1,len(inp_data)))
for i in range(1,len(inp_data)):
    information             = pd.Series(inp_data[i].split('\t')).str.replace(';', "")
    information             = information.str.strip()
    information.index       = pipe_columns
    pipe_information[i]     = information
pipe_information.columns                                = pipe_information.loc['ID',:]
pipe_information                                        = (pipe_information.drop(index='ID')).T
pipe_information['Install_year']                        = pipe_information['Description'].str[-4:]
minimum_install_year                                    = pipe_information.loc[pipe_information['Install_year'] != 'None','Install_year'].min()
pipe_information['Install_year']                        = (pipe_information['Install_year'].str.replace('None',minimum_install_year)).astype(int)
pipe_information[['Length', 'Diameter', 'Roughness']]   = pipe_information[['Length', 'Diameter', 'Roughness']].astype(float)
# Change pipe length unit to kilometers
pipe_information['Length']                              = pipe_information['Length']/1000  
pipe_information['Age']                                 = year_calibrate-pipe_information['Install_year']
#%% Calculate weight of emitter coefficient
weight_emitter_data         = pd.DataFrame()
junction_name               = wn.junction_name_list                             
total_weight_emitter        = (pipe_information['Length']*(1+C_age*pipe_information['Age'])).sum()
pipe_information['W_j']     = 0

for j in range(0,len(pipe_information)):
    Age_of_pipe                 = pipe_information.iloc[j,9]                # 9 means Age column
    Length_pipe                 = pipe_information.iloc[j,2]                # 2 means Lenght column
    pipe_information.iloc[j,10] = (Length_pipe/2)*(1+C_age*Age_of_pipe)     # 10 means W_j column
for k in range(0,len(junction_name)):
    weight_emitter_data.loc[junction_name[k],'W_j']     = pipe_information.loc[(pipe_information['Node1']==junction_name[k]) | (pipe_information['Node2']==junction_name[k]) ,'W_j'].sum()
weight_emitter_data['Wnj']                              = weight_emitter_data['W_j']/total_weight_emitter
#%% Set initial emitter coefficient
initial_c = weight_emitter_data['Wnj']/3600
junc_c    = list(initial_c.index)
for j in range(0,len(junc_c)):
    junc                        = wn.get_node(junc_c[j])
    junc._emitter_coefficient   = initial_c[junc_c[j]]
wn.write_inpfile(inp_after_calibration)
#%% rearrange reservoir flow measurement data
reservoir_measurement_value      = pd.read_table(flow_res_obs,header=None,names=['Name','time','F_m'])
for l in range(0,len(reservoir_measurement_value)):
    if str(reservoir_measurement_value.loc[l,'Name']) != str(np.nan):
        Res_name                 = reservoir_measurement_value.loc[l,'Name']
    elif str(reservoir_measurement_value.loc[l,'Name']) == str(np.nan):
       reservoir_measurement_value.loc[l,'Name'] = Res_name
Reservoir_point                  = reservoir_measurement_value['Name'].drop_duplicates().reset_index(drop=True)
Reservoir_m                      = pd.DataFrame(columns=Reservoir_point,index=range(0,24))
for m in range(0,len(Reservoir_point)):
    Reservoir_m.loc[:,Reservoir_point[m]] = reservoir_measurement_value.loc[reservoir_measurement_value['Name']==Reservoir_point[m],'F_m'].reset_index(drop=True)
Sum_res_m = Reservoir_m.sum(axis=1)
#%% rearrange pressure measurement value
pressure_measurement_value                      = pd.read_table(inlet_press_obs,header=None,names=['Name','time','P_m'])
for p in range(0,len(pressure_measurement_value)):
    if str(pressure_measurement_value.loc[p,'Name']) != str(np.nan):
        point_name                              = pressure_measurement_value.loc[p,'Name']
    elif str(pressure_measurement_value.loc[p,'Name']) == str(np.nan):
       pressure_measurement_value.loc[p,'Name'] = point_name
inlet_junc_point                                = pressure_measurement_value['Name'].drop_duplicates().reset_index(drop=True)
inlet_junc_point                                = [i for i in inlet_junc_point if i in list(wn.junction_name_list)]
pressure_inlet_m                                 = pd.DataFrame(columns=inlet_junc_point,index=range(0,24))
for m in range(0,len(inlet_junc_point )):
    pressure_inlet_m.loc[:,inlet_junc_point [m]] = pressure_measurement_value.loc[pressure_measurement_value['Name']==inlet_junc_point[m],'P_m'].reset_index(drop=True) 
#%% rearrange pressure measurement value
pressure_measurement_value                      = pd.read_table(p_obs,header=None,names=['Name','time','P_m'])
for p in range(0,len(pressure_measurement_value)):
    if str(pressure_measurement_value.loc[p,'Name']) != str(np.nan):
        point_name                              = pressure_measurement_value.loc[p,'Name']
    elif str(pressure_measurement_value.loc[p,'Name']) == str(np.nan):
       pressure_measurement_value.loc[p,'Name'] = point_name
pressure_point                                  = pressure_measurement_value['Name'].drop_duplicates().reset_index(drop=True)
pressure_point                                  = [i for i in pressure_point if i in list(wn.junction_name_list)]
pressure_m                                      = pd.DataFrame(columns=pressure_point,index=range(0,24))
for m in range(0,len(pressure_point)):
    pressure_m.loc[:,pressure_point[m]] = pressure_measurement_value.loc[pressure_measurement_value['Name']==pressure_point[m],'P_m'].reset_index(drop=True)
#%% rearrange flow measurement data
flow_measurement_value      = pd.read_table(f_obs,header=None,names=['Name','time','F_m'])
for l in range(0,len(flow_measurement_value)):
    if str(flow_measurement_value.loc[l,'Name']) != str(np.nan):
        point_name                 = flow_measurement_value.loc[l,'Name']
    elif str(flow_measurement_value.loc[l,'Name']) == str(np.nan):
       flow_measurement_value.loc[l,'Name'] = point_name
flow_point                  = flow_measurement_value['Name'].drop_duplicates().reset_index(drop=True)
flow_point                  = [i for i in flow_point if i in list(wn.link_name_list)]
flow_m                      = pd.DataFrame(columns=flow_point,index=range(0,24))
for m in range(0,len(flow_point)):
    flow_m.loc[:,flow_point[m]] = flow_measurement_value.loc[flow_measurement_value['Name']==flow_point[m],'F_m'].reset_index(drop=True)
flow_observe = abs(flow_m)
#%% get dm use default pattern
junction                            = wn.junction_name_list
detail_junction_pattern             = pd.DataFrame(index=junction,columns=['pattern_name','base_demand'])
for i in junction:
    get_junc                                          = wn.get_node(i)
    pattern_name                                      = get_junc.demand_timeseries_list.pattern_list()[0]
    base_demand                                       = get_junc.base_demand*3600
    if pattern_name != None:
        detail_junction_pattern.loc[i,'pattern_name'] = pattern_name.name
        detail_junction_pattern.loc[i,'base_demand']  = base_demand
junction_use_defulat_pattern   = detail_junction_pattern[(detail_junction_pattern['pattern_name']=='DEFAULTFLOW-F')\
                                                         & (detail_junction_pattern['base_demand']!=0)]
dm_defaultPAT                  = junction_use_defulat_pattern.index
#%% read pipe imformation
pipe_information  = pd.read_excel(pipe,index_col=0)
all_type          = pipe_information['Type'].unique()
#%% Run model before calibration
wn                                  = wntr.network.WaterNetworkModel(inp_after_calibration)

sim                         = wntr.sim.EpanetSimulator(wn)
results                     = sim.run_sim(save_hyd=False)

res_model_before           = abs((results.link['flowrate'][Reservoir_point]*3600).astype(float))
res_model_before.index     = res_model_before.index//3600

junc_p_model_before         = (results.node['pressure'][inlet_junc_point]).astype(float)
junc_p_model_before.index   = junc_p_model_before.index//3600

rmse_flow_before            = np.sqrt(((res_model_before-Reservoir_m)**2).sum().sum()/Reservoir_m.count().sum())
rmse_pressure_before        = np.sqrt(((junc_p_model_before-pressure_inlet_m )**2).sum().sum()/pressure_inlet_m.count().sum())

Sum_res_c       = res_model_before.sum(axis=1)
Sum_res_c.index = Sum_res_c.index//3600

print ('RMSE flow before calibrtion :  '+str(rmse_flow_before)+"\n"
       'RMSE pressure before calibrtion :  '+str(rmse_pressure_before))
#%% function find optimal emitter coefficient
def find_emitter_coeff(x):
    new_emitter_coeff   = x*emitter_coeff
    for j in range(0,len(emitter_junction)):
        junc                        = wn.get_node(emitter_junction[j])
        junc._emitter_coefficient   = new_emitter_coeff[j]/3600
    sim                     = wntr.sim.EpanetSimulator(wn)
    results                 = sim.run_sim(save_hyd=False)
    res_optimize            = abs((results.link['flowrate'][Reservoir_point]*3600).astype(float))
    res_optimize.index      = res_optimize.index//3600
    pjunc_optimize          = (results.node['pressure'][inlet_junc_point]).astype(float)
    pjunc_optimize.index    = pjunc_optimize.index//3600
    rmse_res_optimize       = np.sqrt(((res_optimize-Reservoir_m)**2).sum().sum()/Reservoir_m.count().sum())
    rmse_pjunc_optimize     = np.sqrt(((pjunc_optimize-pressure_inlet_m)**2).sum().sum()/pressure_inlet_m.count().sum())    
    rmse                    = rmse_res_optimize+(rmse_pjunc_optimize)
    return rmse
#%% function find optimal valve setting
def valve_setting_obj(x):
    setting_optimize    = [x[n] for n in range(0,len(valve_optimize))]
    for j in range(0, len(valve_optimize)):
        Valve_NewSetting                            = wn.get_link(valve_optimize[j])
        Valve_NewSetting.initial_setting            = setting_optimize[j]
    sim                     = wntr.sim.EpanetSimulator(wn)
    results                 = sim.run_sim(save_hyd=False)
    flow_optimize           = abs((results.link['flowrate'][flow_point]*3600).astype(float))
    flow_optimize.index     = flow_optimize.index//3600
    pressure_optimize       = (results.node['pressure'][pressure_point]).astype(float)
    pressure_optimize.index = pressure_optimize.index//3600
    rmse_flow_optimize      = np.sqrt(((flow_optimize-flow_m)**2).sum().sum()/flow_m.count().sum())
    rmse_pressure_optimize  = np.sqrt(((pressure_optimize-pressure_m)**2).sum().sum()/pressure_m.count().sum())    
    rmse                    = rmse_flow_optimize+(rmse_pressure_optimize*50)
    return rmse
#%% function find optimal basedemand of dm default pattern
def optimal_basedemand(x):
    old_bd = junction_use_defulat_pattern['base_demand']
    mul    = [x[n] for n in range(0,len(dm_defaultPAT))]
    mul    = pd.Series(mul,index=dm_defaultPAT)
    new_basedemand_series = mul*old_bd
    for j in range(0, len(dm_defaultPAT)):
        Add_new_basedemand                                                   = wn.get_node(dm_defaultPAT[j])
        Add_new_basedemand.demand_timeseries_list[0].base_value              = new_basedemand_series[dm_defaultPAT[j]]/3600
    sim                     = wntr.sim.EpanetSimulator(wn)
    results                 = sim.run_sim(save_hyd=False)
    flow_optimize           = abs((results.link['flowrate'][flow_point]*3600).astype(float))
    flow_optimize.index     = flow_optimize.index//3600
    pressure_optimize       = (results.node['pressure'][pressure_point]).astype(float)
    pressure_optimize.index = pressure_optimize.index//3600
    rmse_flow_optimize      = np.sqrt(((flow_optimize-flow_m)**2).sum().sum()/flow_m.count().sum())
    rmse_pressure_optimize  = np.sqrt(((pressure_optimize-pressure_m)**2).sum().sum()/pressure_m.count().sum())    
    rmse                    = rmse_flow_optimize+(rmse_pressure_optimize*50)     
    return rmse 
#%% function find optimal roughness
def roughness(x):
    wn                                  = wntr.network.WaterNetworkModel(inp_after_calibration)
    wn.options.time.duration            = 23*3600
    wn.options.time.hydraulic_timestep  = 3600
    wn.options.time.report_timestep     = 3600
    mul = [x[n] for n in range(0,len(all_type))]
    for i in range(0,len(all_type)):
        list_pipe = list(pipe_information[pipe_information['Type']==all_type[i]].index)
        for j in range(0,len(list_pipe)):
            pipe_rough              = wn.get_link(list_pipe[j])
            pipe_rough.roughness    = pipe_rough.roughness*mul[i]
    sim                     = wntr.sim.EpanetSimulator(wn)
    results                 = sim.run_sim(save_hyd=False)
    flow_optimize           = abs((results.link['flowrate'][flow_point]*3600).astype(float))
    flow_optimize.index     = flow_optimize.index//3600
    pressure_optimize       = (results.node['pressure'][pressure_point]).astype(float)
    pressure_optimize.index = pressure_optimize.index//3600
    rmse_flow_optimize      = np.sqrt(((flow_optimize-flow_m)**2).sum().sum()/flow_m.count().sum())
    rmse_pressure_optimize  = np.sqrt(((pressure_optimize-pressure_m)**2).sum().sum()/pressure_m.count().sum())    
    rmse                    = rmse_flow_optimize+(rmse_pressure_optimize*50)
    return rmse
#%%
def setting_reservoir_new_emitter(x):
    setting                     = [x[n] for n in range(0,len(Reservoir_point))]
    multiplier                  = x[len(Reservoir_point)]
    new_emitter_coeff           = multiplier*emitter_coeff
    for i in range(0,len(Reservoir_point)):
        valve                       = wn.get_link(Reservoir_point[i])
        valve.initial_setting       = setting[i] 
    for j in range(0,len(emitter_junction)):
        junc                        = wn.get_node(emitter_junction[j])
        junc._emitter_coefficient   = new_emitter_coeff[j]/3600 
    sim                     = wntr.sim.EpanetSimulator(wn)
    results                 = sim.run_sim(save_hyd=False)
    res_optimize            = abs((results.link['flowrate'][Reservoir_point]*3600).astype(float))
    res_optimize.index      = res_optimize.index//3600
    pjunc_optimize          = (results.node['pressure'][inlet_junc_point]).astype(float)
    pjunc_optimize.index    = pjunc_optimize.index//3600
    rmse_res_optimize       = np.sqrt(((res_optimize-Reservoir_m)**2).sum().sum()/Reservoir_m.count().sum())
    rmse_pjunc_optimize     = np.sqrt(((pjunc_optimize-pressure_inlet_m)**2).sum().sum()/pressure_inlet_m.count().sum())    
    rmse                    = rmse_res_optimize+(rmse_pjunc_optimize*200)
    return rmse
#%% read inp file to rearrange emitter coefficient
text                = open(inp_after_calibration,'r')                                
text_data           = text.readlines()
text.close()                                                             
emitters_index      = text_data.index("[EMITTERS]\n")   
quality_index       = text_data.index("[QUALITY]\n")
emitter_coeff       = text_data[(emitters_index+2):(quality_index-1)]
emitter_coeff       = pd.Series(emitter_coeff).str.split(expand=True)
emitter_junction    = list(emitter_coeff[0])
emitter_coeff       = emitter_coeff[1].astype(float) 
#%% Start optimize model by SIV
for loop in range(0,2):
    for iteration in range(0,max_iteration):
        optimize_emitter    = minimize_scalar(find_emitter_coeff,bounds=(0,2000), method='bounded',options={'xatol':1e-04,'disp':3})
    
        new_emitter_coeff   = optimize_emitter.x*emitter_coeff
        for j in range(0,len(emitter_junction)):
            junc                        = wn.get_node(emitter_junction[j])
            junc._emitter_coefficient   = new_emitter_coeff[j]/3600
        wn.write_inpfile(inp_after_calibration)
        
        wn                                  = wntr.network.WaterNetworkModel(inp_after_calibration)
        sim                                 = wntr.sim.EpanetSimulator(wn) 
        results                             = sim.run_sim()
        res_flow_model_set_emitter          = results.node['demand'][Reservoir_point]*-3600
        
        Sum_res_c_emitter       = res_flow_model_set_emitter.sum(axis=1)
        Sum_res_c_emitter.index =  Sum_res_c_emitter .index//3600
        
        pattern_res_c                             = Sum_res_c_emitter/np.mean(Sum_res_c_emitter)
        pattern_res_m                             = Sum_res_m/np.mean(Sum_res_m)
        delta_pattern                             = pattern_res_m-pattern_res_c
        adjusted_pattern_value                    = (delta_pattern-np.mean(delta_pattern))
        
        defualt_pattern                           = pd.Series(wn.get_pattern('DEFAULTFLOW-F').multipliers)
        new_defualt_pattern                       = defualt_pattern+adjusted_pattern_value  
        new_defualt_pattern                       = np.maximum(new_defualt_pattern,0)     
        new_defualt_pattern                       = new_defualt_pattern/np.mean(new_defualt_pattern)
        
        pt                                        = wn.get_pattern('DEFAULTFLOW-F')
        pt.multipliers                            = np.array(new_defualt_pattern)
        wn.write_inpfile(inp_after_calibration)
        
        wn                                  = wntr.network.WaterNetworkModel(inp_after_calibration)
    
        sim                                 = wntr.sim.EpanetSimulator(wn) 
        results                             = sim.run_sim()
        res_flow_model_patt                 = results.node['demand'][Reservoir_point]*-3600
        Sum_res_c_patt                      = res_flow_model_patt.sum(axis=1)
        Sum_res_c_patt.index                =  Sum_res_c_patt.index//3600
        
        text                = open(inp_after_calibration,'r')                                
        text_data           = text.readlines()
        text.close()                                                             
        emitters_index      = text_data.index("[EMITTERS]\n")   
        quality_index       = text_data.index("[QUALITY]\n")
        emitter_coeff       = text_data[(emitters_index+2):(quality_index-1)]
        emitter_coeff       = pd.Series(emitter_coeff).str.split(expand=True)
        emitter_junction    = list(emitter_coeff[0])
        emitter_coeff       = emitter_coeff[1].astype(float) 
        
        plt.figure()
        plt.plot(Sum_res_m,label='Sum flow measurement')
        plt.plot(Sum_res_c_patt,label='Sum flow model')
        plt.legend()
        plt.grid()
        plt.show()
    #%%Set elevation 3:00
    # Set elevation of inlet junction
    p_cal_ele       = (results.node['pressure'][inlet_junc_point]).astype(float)
    p_cal_ele.index = p_cal_ele.index//3600
    ele             = p_cal_ele.loc[3,:]-pressure_inlet_m.loc[3,:]
    for name in inlet_junc_point:
        junc                = wn.get_node(name)
        junc.elevation      = junc.elevation+ele[name]
    # Set elevation of pressure measurement junction
    p_cal_ele       = (results.node['pressure'][pressure_point]).astype(float)
    p_cal_ele.index = p_cal_ele.index//3600
    ele             = p_cal_ele.loc[3,:]-pressure_m.loc[3,:]
    for name in pressure_point:
        junc                = wn.get_node(name)
        junc.elevation      = junc.elevation+ele[name]
    wn.write_inpfile(inp_after_calibration)
    wn  = wntr.network.WaterNetworkModel(inp_after_calibration)
    #%% Start optimize model by basedemand of dm default pattern
    # bound                                           = pd.DataFrame(index=junction_use_defulat_pattern.index)
    # bound['Lower']                                  = 0.5
    # bound['Upper']                                  = 2
    # bound                                           = np.array(bound)
    
    # plt.figure()
    # algorithm_param = {'max_num_iteration':200,\
    #                     'population_size':10,\
    #                     'mutation_probability':0.1,\
    #                     'elit_ratio': 0.01,\
    #                     'crossover_probability': 0.5,\
    #                     'parents_portion': 0.3,\
    #                     'crossover_type':'uniform',\
    #                     'max_iteration_without_improv':None}
        
    # model=ga(function=optimal_basedemand,dimension=len(dm_defaultPAT),variable_type='real',variable_boundaries=bound,\
    #           function_timeout=3600,algorithm_parameters=algorithm_param)
    # model.run()
    # old_bd = junction_use_defulat_pattern['base_demand']
    # mul    = model.best_variable
    # mul    = pd.Series(mul,index=dm_defaultPAT)
    # new_basedemand_series = mul*old_bd
    # for j in range(0, len(dm_defaultPAT)):
    #     Add_new_basedemand                                                   = wn.get_node(dm_defaultPAT[j])
    #     Add_new_basedemand.demand_timeseries_list[0].base_value              = new_basedemand_series[dm_defaultPAT[j]]/3600
    # wn.write_inpfile(inp_after_calibration)
    # wn = wntr.network.WaterNetworkModel(inp_after_calibration) 
    # sim     = wntr.sim.EpanetSimulator(wn) 
    # results = sim.run_sim()
    #%%
    plt.figure()
    bound                               = np.ones((len(Reservoir_point)+1,2))
    bound[0:len(Reservoir_point),0]    *= 0
    bound[0:len(Reservoir_point),1]    *= 5
    bound[len(Reservoir_point),1]      *= 2
    algorithm_param = {'max_num_iteration':200,\
                        'population_size':5,\
                        'mutation_probability':0.1,\
                        'elit_ratio': 0.01,\
                        'crossover_probability': 0.5,\
                        'parents_portion': 0.3,\
                        'crossover_type':'uniform',\
                        'max_iteration_without_improv':None}
    model    = ga(function=setting_reservoir_new_emitter,dimension=len(Reservoir_point)+1,variable_type='real',variable_boundaries=bound,\
               function_timeout=3600,algorithm_parameters=algorithm_param)
    model.run()
    setting                     = model.best_variable[0:len(Reservoir_point)]
    for i in range(0,len(Reservoir_point)):
        valve                   = wn.get_link(Reservoir_point[i])
        valve.initial_setting   = setting[i] 
    
    multiplier                  = model.best_variable[len(Reservoir_point)]
    new_emitter_coeff           = multiplier*emitter_coeff
    for j in range(0,len(emitter_junction)):
        junc                        = wn.get_node(emitter_junction[j])
        junc._emitter_coefficient   = new_emitter_coeff[j]/3600
    wn.write_inpfile(inp_after_calibration)
    wn = wntr.network.WaterNetworkModel(inp_after_calibration) 
    sim     = wntr.sim.EpanetSimulator(wn) 
    results = sim.run_sim()
    
    text                = open(inp_after_calibration,'r')                                
    text_data           = text.readlines()
    text.close()                                                             
    emitters_index      = text_data.index("[EMITTERS]\n")   
    quality_index       = text_data.index("[QUALITY]\n")
    emitter_coeff       = text_data[(emitters_index+2):(quality_index-1)]
    emitter_coeff       = pd.Series(emitter_coeff).str.split(expand=True)
    emitter_junction    = list(emitter_coeff[0])
    emitter_coeff       = emitter_coeff[1].astype(float)
    #%%Set elevation 3:00
    #Set elevation of inlet junction
    p_cal_ele       = (results.node['pressure'][inlet_junc_point]).astype(float)
    p_cal_ele.index = p_cal_ele.index//3600
    ele             = p_cal_ele.loc[3,:]-pressure_inlet_m.loc[3,:]
    for name in inlet_junc_point:
        junc                = wn.get_node(name)
        junc.elevation      = junc.elevation+ele[name]
    # Set elevation of pressure measurement junction
    p_cal_ele       = (results.node['pressure'][pressure_point]).astype(float)
    p_cal_ele.index = p_cal_ele.index//3600
    ele             = p_cal_ele.loc[3,:]-pressure_m.loc[3,:]
    for name in pressure_point:
        junc                = wn.get_node(name)
        junc.elevation      = junc.elevation+ele[name]
    wn.write_inpfile(inp_after_calibration)
    wn  = wntr.network.WaterNetworkModel(inp_after_calibration)
    #%%Start optimize model by roughness
    bound                        = np.ones((len(all_type),2))
    bound[0:len(all_type),0]    *= 0.8
    bound[0:len(all_type),1]    *= 1
    plt.figure()
    algorithm_param = {'max_num_iteration':200,\
                        'population_size':5,\
                        'mutation_probability':0.1,\
                        'elit_ratio': 0.01,\
                        'crossover_probability': 0.5,\
                        'parents_portion': 0.3,\
                        'crossover_type':'uniform',\
                        'max_iteration_without_improv':None}
    model    = ga(function=roughness,dimension=len(all_type),variable_type='real',variable_boundaries=bound,\
               function_timeout=3600,algorithm_parameters=algorithm_param)
    model.run()
    wn                                  = wntr.network.WaterNetworkModel(inp_after_calibration)
    wn.options.time.duration            = 23*3600
    wn.options.time.hydraulic_timestep  = 3600
    wn.options.time.report_timestep     = 3600
    mul = model.best_variable
    for i in range(0,len(all_type)):
        list_pipe = list(pipe_information[pipe_information['Type']==all_type[i]].index)
        for j in range(0,len(list_pipe)):
            pipe_rough              = wn.get_link(list_pipe[j])
            pipe_rough.roughness    = pipe_rough.roughness*mul[i]
    wn.write_inpfile(inp_after_calibration)
    wn  = wntr.network.WaterNetworkModel(inp_after_calibration)
    sim     = wntr.sim.EpanetSimulator(wn) 
    results = sim.run_sim()
    #%%Set elevation 3:00
    # Set elevation of inlet junction
    p_cal_ele       = (results.node['pressure'][inlet_junc_point]).astype(float)
    p_cal_ele.index = p_cal_ele.index//3600
    ele             = p_cal_ele.loc[3,:]-pressure_inlet_m.loc[3,:]
    for name in inlet_junc_point:
        junc                = wn.get_node(name)
        junc.elevation      = junc.elevation+ele[name]
    # Set elevation of pressure measurement junction
    p_cal_ele       = (results.node['pressure'][pressure_point]).astype(float)
    p_cal_ele.index = p_cal_ele.index//3600
    ele             = p_cal_ele.loc[3,:]-pressure_m.loc[3,:]
    for name in pressure_point:
        junc                = wn.get_node(name)
        junc.elevation      = junc.elevation+ele[name]
    wn.write_inpfile(inp_after_calibration)
    wn  = wntr.network.WaterNetworkModel(inp_after_calibration)
    #%% Start optimize model by valve setting
    valve_setting                                   = pd.DataFrame(wn.query_link_attribute("initial_setting").dropna())
    InitialSetting_ValveOptimize                    = valve_setting.loc[(valve_setting[0]>0.189138) & (valve_setting[0]<1580.7),0]
    valve_optimize                                  = list(InitialSetting_ValveOptimize.index)
    valve_optimize                                  = [i for i in valve_optimize if i not in wn.reservoir_name_list]
    setting                                         = pd.DataFrame(InitialSetting_ValveOptimize[valve_optimize])
    setting['%open']                                = (np.log(1580.7)-np.log(setting[0]))/0.090309 
    setting['%Lower']                               = np.maximum(setting['%open']-20,0)
    setting['%Upper']                               = np.minimum(setting['%open']+20,100)           
    setting['Lower']                                = 1580.7*np.exp(-0.090309*setting['%Lower'])
    setting['Upper']                                = 1580.7*np.exp(-0.090309*setting['%Upper'])
    bound                                           = pd.DataFrame()
    bound['Lower']                                  = setting['Upper'].reset_index(drop=True)
    bound['Upper']                                  = setting['Lower'].reset_index(drop=True)
    bound                                           = np.array(bound)
    plt.figure()
    algorithm_param = {'max_num_iteration':200 ,\
                        'population_size':10,\
                        'mutation_probability':0.1,\
                        'elit_ratio': 0.01,\
                        'crossover_probability': 0.5,\
                        'parents_portion': 0.3,\
                        'crossover_type':'uniform',\
                        'max_iteration_without_improv':None}
    model=ga(function=valve_setting_obj,dimension=len(valve_optimize),variable_type='real',variable_boundaries=bound,\
              function_timeout=3600,algorithm_parameters=algorithm_param)
    model.run()
    setting_optimize    = model.best_variable
    for j in range(0, len(valve_optimize)):
        Valve_NewSetting                            = wn.get_link(valve_optimize[j])
        Valve_NewSetting.initial_setting            = setting_optimize[j]
    wn.write_inpfile(inp_after_calibration)
    wn      = wntr.network.WaterNetworkModel(inp_after_calibration)
    sim     = wntr.sim.EpanetSimulator(wn) 
    results = sim.run_sim()
    #%%Set elevation 3:00
    # Set elevation of inlet junction
    p_cal_ele       = (results.node['pressure'][inlet_junc_point]).astype(float)
    p_cal_ele.index = p_cal_ele.index//3600
    ele             = p_cal_ele.loc[3,:]-pressure_inlet_m.loc[3,:]
    for name in inlet_junc_point:
        junc                = wn.get_node(name)
        junc.elevation      = junc.elevation+ele[name]
    # Set elevation of pressure measurement junction
    p_cal_ele       = (results.node['pressure'][pressure_point]).astype(float)
    p_cal_ele.index = p_cal_ele.index//3600
    ele             = p_cal_ele.loc[3,:]-pressure_m.loc[3,:]
    for name in pressure_point:
        junc                = wn.get_node(name)
        junc.elevation      = junc.elevation+ele[name]
    wn.write_inpfile(inp_after_calibration)
    wn  = wntr.network.WaterNetworkModel(inp_after_calibration)