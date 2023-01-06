import wntr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
from shapely.geometry import Point
import pickle
#%% input data
inp_file           = './SAMSEN.inp'
dm_influence_file  = './DM-influenced-bound.csv'
bound_pump_file    = './Bound-of-pumping-station.csv'
station_valve_file = './Open-valves-at-pumping-station.csv'
minimum_pressure   = 10
minimum_DMA        = 5
decision_time      = '23:00'
number_hours       = 4
pumping_station    = ['SAMSEN_DU1','SAMSEN_O1','SAMSEN_T','SAMSEN_X']
save_file          = './For_find_influenced.inp' 
max_head_reservoir = 50 
#%% setting default model and read data
wn                                  = wntr.network.WaterNetworkModel(inp_file)
wn.options.time.duration            = 23*3600
wn.options.time.hydraulic_timestep  = 3600
wn.options.time.report_timestep     = 3600

sim                  = wntr.sim.EpanetSimulator(wn)
results              = sim.run_sim()

total_head = 0 
for r in pumping_station:
    reservoir    = wn.get_node(r)
    total_head  += reservoir.head_timeseries.base_value
max_pattern = max_head_reservoir/(total_head/len(pumping_station))

col_station         = [i.split('_', 1)[0] for i in pumping_station][0]
dm_list             = list(pd.read_csv(dm_influence_file)[col_station].dropna())
bound_pump          = list(pd.read_csv(bound_pump_file)[col_station].dropna())
valve_station       = list(pd.read_csv(station_valve_file)[col_station].dropna())

connect_dm_data     = pd.DataFrame(index=wn.query_link_attribute('start_node').index)
connect_dm_data['Start_node'] = [i.name for i in wn.query_link_attribute('start_node')]
connect_dm_data['End_node'] = [i.name for i in wn.query_link_attribute('end_node')]
connect_dm_data = connect_dm_data.loc[['TCV-'+i for i in dm_list],:]
node_before_dm = list(connect_dm_data['Start_node'])

junc_source = wn.query_node_attribute('base_demand')
junc_source = list(junc_source[junc_source < 0].index)
#%% Build DM GIS
xy    = wn.query_node_attribute('coordinates')
dm    = [i for i in list(wn.junction_name_list) if i[0]=='D']
xy    = xy[dm]
xy_dm = pd.DataFrame(index=xy.index,columns=['x','y','point'])
for i in list(xy.index):
    xy_dm.loc[i,'x']     = xy[i][0]
    xy_dm.loc[i,'y']     = xy[i][1]
    xy_dm.loc[i,'point'] = Point(xy_dm.loc[i,'x'],xy_dm.loc[i,'y'])
dm_gis = {'DM_NAME': xy_dm.index , 'geometry': xy_dm['point']}
dm_gis = gpd.GeoDataFrame(dm_gis, crs="EPSG:32647")

dma  = gpd.read_file('../DMA_GIS/DMA.shp')
pipe = gpd.read_file('../PIPE_GIS/MainPipe_3res.shp')

dma_name         = ['DM-'+i[0:2]+'-'+i[2:4]+'-'+i[4:6] for i in list(dma['BLOCKNAME'])]
dma['BLOCKNAME'] = dma_name

dma_influ =[] 
dm_for_check_size = dm_gis.loc[dm_list]
for d in list(dm_for_check_size.index):
    for a in range(0,len(dma)):
        if dm_for_check_size.loc[d,'geometry'].within(dma.loc[a,'geometry']) == True:
            dma_influ.append(dma.loc[a,'BLOCKNAME'])
dma_influ = pd.Series(dma_influ).drop_duplicates().reset_index(drop=True)
total_loop_compute  = len(dma_influ)//minimum_DMA
num_dma = len(dma_influ)
#%% rearrange time select
decision_time  = int(decision_time.split(sep=':')[0])
dt = decision_time
time_select = []
t = [decision_time]
for i in range(0,number_hours):
    if decision_time+1 > 23:
        decision_time = 0
        time_select.append(decision_time)
        t.append(decision_time)
    else:
        decision_time+=1
        time_select.append(decision_time)
        t.append(decision_time)
decision_time = dt
#%% dataframe for keep results
flow_reservoir         = abs(results.link['flowrate'][valve_station]).astype(float)*3600
flow_reservoir.index   = flow_reservoir.index//3600
flow_reservoir         = flow_reservoir.loc[t,:]
flow_reservoir         = np.round(flow_reservoir.sum().sum(),2)

dm_influence_per_loop  = pd.DataFrame()
dma_influence_per_loop = pd.DataFrame()
pattern_mul_per_loop   = pd.DataFrame(index=[decision_time]+time_select)
dm_new_emitter         = []
dma_check              = []
flow_use               = [flow_reservoir]
#%% Setting valve characteristic
valve_trunk  = wn.valve_name_list
valve_trunk  = [i for i in valve_trunk if i[0:3] == 'VAL']
valve_dm     = list(connect_dm_data.index)

for val in bound_pump:
    valve                = wn.get_link(val)
    valve.initial_status = 'Closed'
    
for val in valve_station:
    valve                 = wn.get_link(val)
    valve.initial_setting = 0

for val in valve_trunk:
    valve                 = wn.get_link(val)
    valve.initial_setting = 0

for val in valve_dm:
    valve                 = wn.get_link(val)
    dia                   = valve.diameter
    wn.remove_link(val)
    wn.add_valve(val,connect_dm_data.loc[val,'Start_node'],connect_dm_data.loc[val,'End_node'],\
                 diameter=dia,valve_type='PRV', setting=minimum_pressure)

wn.write_inpfile(save_file)
inp_file  = save_file
wn        = wntr.network.WaterNetworkModel(inp_file)
#%% Start select DMA by pressure at decision_time
loop = 0
n    = 0 
# for loop in range(0,total_loop_compute):
while n < num_dma:
    if num_dma-n < minimum_DMA:
        minimum_DMA = num_dma-n
    
    head_pattern = []
    count_DMA = 0 
    step_increase = 0.5
    for r in pumping_station:
        res = wn.get_node(r)
        head_pattern.append(res.head_timeseries.pattern_name)
        
    while count_DMA < minimum_DMA:
        mul_print = []
        for h in range(0,len(head_pattern)):
            pat_value  = wn.get_pattern(head_pattern[h])
            pat_value.multipliers[decision_time] = pat_value.multipliers[decision_time]+step_increase
            mul_print.append(pat_value.multipliers[decision_time])
            
        if pat_value.multipliers[decision_time] >= max_pattern:
            break

        sim                  = wntr.sim.EpanetSimulator(wn)
        results              = sim.run_sim()
        pressure_fdm         = (results.node['pressure'][node_before_dm]).astype(float)
        pressure_fdm.index   = pressure_fdm.index//3600
        pressure_fdm         = pressure_fdm.loc[decision_time,:]
            
        if loop == 0:
            node_gain_min_press   = (pressure_fdm[pressure_fdm>= minimum_pressure].index)
            node_gain_min_press   = pd.Series(node_gain_min_press)
            dm_gain_min_press     = connect_dm_data.loc[connect_dm_data['Start_node'].isin(node_gain_min_press),'End_node']
            dm_gain_min_press     = dm_gain_min_press.reset_index(drop=True)
            
        elif loop > 0:
            node_gain_min_press   = (pressure_fdm[pressure_fdm>= minimum_pressure].index)
            node_gain_min_press   = pd.Series(node_gain_min_press)
            dm_gain_min_press     = connect_dm_data.loc[connect_dm_data['Start_node'].isin(node_gain_min_press),'End_node']
            dm_gain_min_press   =  dm_gain_min_press[~dm_gain_min_press.isin(dm_influence_per_loop[loop-1])]
            dm_gain_min_press   = pd.Series(dm_gain_min_press)
            dm_gain_min_press     = dm_gain_min_press.reset_index(drop=True)
        
        if (len(dm_gain_min_press)>0) & (loop == 0):
            step_increase   = 0.1
            check_area = []
            for d in dm_gain_min_press:
                for a in range(0,len(dma)):
                    if dm_gis.loc[d,'geometry'].within(dma.loc[a,'geometry']) == True:
                        check_area.append(dma.loc[a,'BLOCKNAME'])
            dma_gain_min_press  = pd.Series(check_area)
            dma_gain_min_press  = pd.Series(dma_gain_min_press.copy()).drop_duplicates().reset_index(drop=True)
            count_DMA           = len(dma_gain_min_press)
            
        elif (len(dm_gain_min_press)>0) & (loop > 0):
            step_increase   = 0.1
            check_area = []
            for d in dm_gain_min_press:
                for a in range(0,len(dma)):
                    if dm_gis.loc[d,'geometry'].within(dma.loc[a,'geometry']) == True:
                        check_area.append(dma.loc[a,'BLOCKNAME'])
            dma_gain_min_press  = pd.Series(check_area)
            dma_gain_min_press  = pd.Series(dma_gain_min_press.copy()).drop_duplicates().reset_index(drop=True)
            dma_gain_min_press  = dma_gain_min_press[~dma_gain_min_press.isin(dma_check)]
            dma_gain_min_press  = dma_gain_min_press.reset_index(drop=True)
            count_DMA           = len(dma_gain_min_press)
    if pat_value.multipliers[decision_time] >= max_pattern:
        break
    print('#####################  loop : '+str(loop+1)+'  #####################')
    print('Multiplier at '+str(decision_time) +':00'+' : '+str(np.round(mul_print,2)))
    dm_influence_per_loop                     = pd.concat([dm_influence_per_loop,dm_gain_min_press],ignore_index=True,axis=1)
    dma_influence_per_loop                    = pd.concat([dma_influence_per_loop,dma_gain_min_press],ignore_index=True,axis=1)
    pattern_mul_per_loop.loc[decision_time,loop] = str(np.round(mul_print,2))
    dm_new_emitter                            = dm_new_emitter+list(dm_gain_min_press)
    dm_new_emitter                            = pd.Series(dm_new_emitter).drop_duplicates().reset_index(drop=True)
    dm_new_emitter                            = list(dm_new_emitter)
    n += count_DMA
#%% finding hourly pattern in time select 
    step_increase = 0.5
    for i in time_select:
        count_DMA = 0
        while count_DMA < len(dma_gain_min_press):
            mul_print = []
            for h in range(0,len(head_pattern)):
                pat_value  = wn.get_pattern(head_pattern[h])
                pat_value.multipliers[i] = pat_value.multipliers[i]+step_increase
                mul_print.append(pat_value.multipliers[i])
                
            if pat_value.multipliers[i] >= max_pattern:
                dm_influence_per_loop = dm_influence_per_loop.iloc[: , :-1]
                dma_influence_per_loop = dma_influence_per_loop.iloc[: , :-1]
                break    
            sim                  = wntr.sim.EpanetSimulator(wn)
            results              = sim.run_sim()
            pressure_fdm         = (results.node['pressure'][node_gain_min_press]).astype(float)
            pressure_fdm.index   = pressure_fdm.index//3600
            pressure_fdm         = pressure_fdm.loc[i,:]
    
            min_press          = (pressure_fdm[pressure_fdm>= minimum_pressure].index)
            min_press          = connect_dm_data.loc[connect_dm_data['Start_node'].isin(min_press),'End_node']
            min_press          = min_press.reset_index(drop=True)
            
            if (len(min_press)>0) & (loop == 0):
                step_increase  = 0.1
                check_area = []
                for d in min_press:
                    for a in range(0,len(dma)):
                        if dm_gis.loc[d,'geometry'].within(dma.loc[a,'geometry']) == True:
                            check_area.append(dma.loc[a,'BLOCKNAME'])
                dma_min_press  = pd.Series(check_area)
                dma_min_press  = pd.Series(dma_min_press.copy()).drop_duplicates().reset_index(drop=True)
                count_DMA      = len(dma_min_press)
            elif (len(min_press)>0) & (loop > 0):
                step_increase  = 0.1
                check_area = []
                for d in min_press:
                    for a in range(0,len(dma)):
                        if dm_gis.loc[d,'geometry'].within(dma.loc[a,'geometry']) == True:
                            check_area.append(dma.loc[a,'BLOCKNAME'])
                dma_min_press  = pd.Series(check_area)
                dma_min_press  = pd.Series(dma_min_press.copy()).drop_duplicates().reset_index(drop=True)
                dma_min_press  = dma_min_press[~dma_min_press.isin(dma_check)]
                dma_min_press  = dma_min_press.reset_index(drop=True)
                count_DMA      = len(dma_min_press)   
            
        print('Multiplier at '+str(i)+':00' +' : '+str(np.round(mul_print,2)))
        pattern_mul_per_loop.loc[i,loop] = str(np.round(mul_print,2))
        if pat_value.multipliers[i] >= max_pattern:
            pattern_mul_per_loop = pattern_mul_per_loop.iloc[: , :-1]
            break 
    if pat_value.multipliers[i] >= max_pattern:
        break 
    #%%
    # wn.write_inpfile('tt.inp')
    dma_check              = dma_check+list(dma_gain_min_press)
    flow_reservoir         = abs(results.link['flowrate'][valve_station]).astype(float)*3600
    flow_reservoir.index   = flow_reservoir.index//3600
    flow_reservoir         = flow_reservoir.loc[t,:]
    flow_reservoir         = np.round(flow_reservoir.sum().sum(),0)
    print('Flow used' +' : '+str(flow_reservoir)+' CMH')
    flow_use.append(flow_reservoir)
    #%% select junction on trunkmain
    junction_trunk         = [i for i in list(wn.junction_name_list) if (i[0]!='D')&(i not in junc_source)]
    pressure_trunk         = (results.node['pressure'][junction_trunk]).astype(float)
    pressure_trunk.index   = pressure_trunk.index//3600
    pressure_trunk         = pressure_trunk.loc[t,:]
    avg_pressure_trunk     = pressure_trunk.mean()
    junc_trunk_emitter     = list(avg_pressure_trunk[avg_pressure_trunk>=minimum_pressure ].index)
#%% devide emitter coefficient by 2
    wn                                  = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration            = 23*3600
    wn.options.time.hydraulic_timestep  = 3600
    wn.options.time.report_timestep     = 3600
    for dm in dm_new_emitter:
        dm_select                      = wn.get_node(dm)
        if dm_select._emitter_coefficient==None:
            C_coeff = 0
        else:
            C_coeff = dm_select._emitter_coefficient  
        dm_select._emitter_coefficient = C_coeff/2
    
    for junc in junc_trunk_emitter:
        junc_select                      = wn.get_node(junc)
        if junc_select._emitter_coefficient==None:
            C_coeff = 0
        else:
            C_coeff = junc_select._emitter_coefficient
        junc_select._emitter_coefficient = C_coeff/2    
    loop +=1
#%%
dma_accepted = []
for col in dma_influence_per_loop.columns:
    name=dma_influence_per_loop[col].dropna()
    dma_accepted+=list(name)
dma_not_accepted = [n for n in list(dma_influ) if n not in dma_accepted]
#%% find dma no inlet
dma_empty = list(pd.read_csv('DMA_no_inlet.csv',index_col=0)['0'])

cheak_near = dma_influ.str[0:8]
cheak_near = cheak_near.drop_duplicates().reset_index(drop=True)

dma_empty = [i for i in dma_empty if i[0:8] in list(cheak_near)]
dma_empty = dma[dma['BLOCKNAME'].isin(dma_empty)].reset_index(drop=True)
dma_empty['DMA_nearest'] = 0
dma_empty['Dist']=0

dma_influ = dma[dma['BLOCKNAME'].isin(list(dma_influ))].reset_index(drop=True)
#%%
for i in range(0,len(dma_empty)):
    poly_1     = dma_empty.iloc[i,:].geometry
    dist_check = 0
    for j in range(0,len(dma_influ)):
        poly_2   = dma_influ.iloc[j,:].geometry
        dist_cal = poly_1.distance(poly_2)
        if j==0:
            dist_check          = dist_cal
            dma_empty.iloc[i,8] = dma_influ.iloc[j,4]
            dma_empty.iloc[i,9] = dist_cal
        elif (j>0) & (dist_cal<dist_check):
            dist_check          = dist_cal
            dma_empty.iloc[i,8] = dma_influ.iloc[j,4]  
            dma_empty.iloc[i,9] = dist_check
dma_empty = dma_empty[dma_empty['Dist']<=1000]
#%%
head_per_loop = pattern_mul_per_loop.copy()
for col in head_per_loop.columns:
    head_per_loop[col] = head_per_loop[col].str.replace("[", "")
    head_per_loop[col] = head_per_loop[col].str.replace("]", "")
    val                = head_per_loop[col].str.split(" ",expand=True).fillna(0)
    val[val==""]       = 0
    head_per_loop[col] = val.astype(float).mean(axis=1)

head_per_loop=head_per_loop.astype(float)
head_per_loop*=(total_head/len(pumping_station))
head_max=np.round(head_per_loop.max(),1)
head_avg=np.round(head_per_loop.mean(),1)
#%%
num = 0 
dm_new_emitter   = []
recheck = []
col_head=[]
dma_impossible_per_loop = pd.DataFrame()
increase = np.arange(50,105,5,dtype=int)
increase = increase/(total_head/len(pumping_station))

for i in range(0,len(increase)):
    if num == len(dma_not_accepted):
        break
# for i in range(0,5):   
    for h in range(0,len(head_pattern)):
        pat_value  = wn.get_pattern(head_pattern[h])
        pat_value.multipliers[decision_time] = increase[i]
    sim                  = wntr.sim.EpanetSimulator(wn)
    results              = sim.run_sim()
    pressure_fdm         = (results.node['pressure'][node_before_dm]).astype(float)
    pressure_fdm.index   = pressure_fdm.index//3600
    pressure_fdm         = pressure_fdm.loc[decision_time,:]
    
    node_gain_min_press   = (pressure_fdm[pressure_fdm>= minimum_pressure].index)
    node_gain_min_press   = pd.Series(node_gain_min_press)
    dm_gain_min_press     = connect_dm_data.loc[connect_dm_data['Start_node'].isin(node_gain_min_press),'End_node']
    dm_gain_min_press     = dm_gain_min_press.reset_index(drop=True)
    
    dm_new_emitter                            = dm_new_emitter+list(dm_gain_min_press)
    dm_new_emitter                            = pd.Series(dm_new_emitter).drop_duplicates().reset_index(drop=True)
    dm_new_emitter                            = list(dm_new_emitter)
    
    pressure_trunk         = (results.node['pressure'][junction_trunk]).astype(float)
    pressure_trunk.index   = pressure_trunk.index//3600
    pressure_trunk         = pressure_trunk.loc[t,:]
    avg_pressure_trunk     = pressure_trunk.mean()
    junc_trunk_emitter     = list(avg_pressure_trunk[avg_pressure_trunk>=minimum_pressure ].index)
    
    dma_residue   =  dma[dma['BLOCKNAME'].isin(dma_not_accepted)].reset_index(drop=True)
    dma_impossible = []
    for d in dm_gain_min_press:
        for a in range(0,len(dma_residue)):
            if dm_gis.loc[d,'geometry'].within(dma_residue.loc[a,'geometry']) == True:
                dma_impossible.append(dma_residue.loc[a,'BLOCKNAME'])
    # if i == 0:     
    #     if len(dma_impossible)>0:
    #         dma_impossible_per_loop=pd.concat([dma_impossible_per_loop,pd.DataFrame(dma_impossible)],ignore_index=True,axis=1)
    #         col_count+=1
    #         num+=len(dma_impossible)
    # if i > 0:
    dma_impossible = [n for n in dma_impossible if n not in recheck]
    dma_impossible = pd.Series(dma_impossible).drop_duplicates().reset_index(drop=True)
    dma_impossible = list(dma_impossible)
    if len(dma_impossible)>0:
        dma_impossible_per_loop=pd.concat([dma_impossible_per_loop,pd.DataFrame(dma_impossible)],ignore_index=True,axis=1)
        num+=len(dma_impossible)
        col_head.append(int(increase[i]*(total_head/len(pumping_station))))
    recheck+=dma_impossible
    
    wn                                  = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration            = 23*3600
    wn.options.time.hydraulic_timestep  = 3600
    wn.options.time.report_timestep     = 3600
    for dm in dm_new_emitter:
        dm_select                      = wn.get_node(dm)
        if dm_select._emitter_coefficient==None:
            C_coeff = 0
        else:
            C_coeff = dm_select._emitter_coefficient  
        dm_select._emitter_coefficient = C_coeff/2
    
    for junc in junc_trunk_emitter:
        junc_select                      = wn.get_node(junc)
        if junc_select._emitter_coefficient==None:
            C_coeff = 0
        else:
            C_coeff = junc_select._emitter_coefficient
        junc_select._emitter_coefficient = C_coeff/2  
dma_not_accepted=[i for i in dma_not_accepted if i not in recheck]
dma_impossible_per_loop.columns=col_head
#%% plot result
df_conclusion = pd.concat([dma_influence_per_loop,dma_impossible_per_loop],ignore_index=True,axis=1)

fig, ax             = plt.subplots(figsize=(16,9))
ax.set_title('Flow used (Q) before pressure increased : '+str(flow_use[0])+' CMH')
dma.plot(ax=ax,cmap="viridis",alpha=0.2,linewidth=1,edgecolor='black')
# dm.plot(ax=ax,color='black',markersize=5)
pipe.plot(ax=ax,color='blue')

color_plot = cm.rainbow(np.linspace(0,1,len(df_conclusion.columns)))
for f in range(0,len(df_conclusion.columns)): 
    
    dma_influence    =  dma[dma['BLOCKNAME'].isin(df_conclusion[f])]
    dma_influence.plot(ax=ax,color=color_plot[f])
    dma_influence['coords'] = dma_influence['geometry'].apply(lambda x: x.representative_point().coords[:])
    dma_influence['coords'] = [coords[0] for coords in dma_influence['coords']]
    for idx, row in dma_influence.iterrows():
        plt.annotate(s=row['BLOCKNAME'], xy=row['coords'],
                      horizontalalignment='center')
       
    dma_influence    =  dma_empty[dma_empty['DMA_nearest'].isin(df_conclusion[f])]
    dma_influence.plot(ax=ax,color=color_plot[f])
    dma_influence['coords'] = dma_influence['geometry'].apply(lambda x: x.representative_point().coords[:])
    dma_influence['coords'] = [coords[0] for coords in dma_influence['coords']]
    for idx, row in dma_influence.iterrows():
        plt.annotate(s=row['BLOCKNAME'], xy=row['coords'],
                      horizontalalignment='center')

dma_miss    =  dma[dma['BLOCKNAME'].isin(dma_not_accepted)]
dma_miss.plot(ax=ax,color='gray')        
dma_miss['coords'] = dma_miss['geometry'].apply(lambda x: x.representative_point().coords[:])
dma_miss['coords'] = [coords[0] for coords in dma_miss['coords']]
for idx, row in dma_miss.iterrows():
    plt.annotate(s=row['BLOCKNAME'], xy=row['coords'],
                  horizontalalignment='center')   
    
dma_miss    =  dma_empty[dma_empty['DMA_nearest'].isin(dma_not_accepted)]
dma_miss.plot(ax=ax,color='gray')        
dma_miss['coords'] = dma_miss['geometry'].apply(lambda x: x.representative_point().coords[:])
dma_miss['coords'] = [coords[0] for coords in dma_miss['coords']]
for idx, row in dma_miss.iterrows():
    plt.annotate(s=row['BLOCKNAME'], xy=row['coords'],
                  horizontalalignment='center')   

dm_gis.loc[dm_list,:].plot(ax=ax,color='green')
l ={}
c = 0
for f in range(0,len(df_conclusion.columns)):
    if f < len(dma_influence_per_loop.columns):
        l[f]= mpatches.Patch(color=color_plot[f], label=str(f+1)+' Q : '+str(flow_use[f+1])+\
                              ', $H_{max}$ : '+str(head_max[f])+', $H_{avg}$ : '+str(head_avg[f]))
    else:
        l[f]= mpatches.Patch(color=color_plot[f], label='$H_{decision}$ : '+str(col_head[c]))
        c+=1
l[f+1]= mpatches.Patch(color='gray', label='DMA below condition')

ax.legend(handles=[l[f] for f in range(0,len(df_conclusion.columns)+1)],bbox_to_anchor=[1, 1], loc='upper left')
#%%
below = pd.DataFrame({'below condition':dma_not_accepted})
writer = pd.ExcelWriter(col_station+'.xlsx')
# write dataframe to excel sheet named 'marks'
dma_influence_per_loop.to_excel(writer, 'dma_influence')
dma_impossible_per_loop.to_excel(writer, 'dma_over_pump_capacity')
below.to_excel(writer, 'dma_below_condition')
pattern_mul_per_loop.to_excel(writer, 'pattern')
# save the excel file
writer.save()
pickle.dump(fig,open((col_station+'.fig.pickle'),'wb'))