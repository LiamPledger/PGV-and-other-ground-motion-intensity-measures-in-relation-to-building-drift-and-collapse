import openseespy.opensees as ops
import numpy as np
import importlib
import pandas as pd
from SmartAnalyze import SmartAnalyzeTransient
import openseespy.postprocessing.Get_Rendering as opsplt
import sys


def IDA(gmno, direction, modelno, SAscaled, curdir, curdir_out, foldername, gmfolderloc, dlim, Sa_perturb, dt_redfac):
    # wipe previously existing models
    ops.wipe()

    
    # Get the model ID based on the provided model number index
    # model_ID_list = ["RCW_5S_10DL", "RCW_5S_15DL", "RCW_5S_20DL", "RCW_5S_25DL",
    #                  "RCW_10S_10DL", "RCW_10S_15DL", "RCW_10S_20DL", "RCW_10S_25DL",
    #                  "RCW_15S_10DL", "RCW_15S_15DL", "RCW_15S_20DL", "RCW_15S_25DL",
    #                  "RCW_20S_10DL", "RCW_20S_15DL", "RCW_20S_20DL", "RCW_20S_25DL"]
    
    model_ID_list = ["RCW_20S_10DL", "RCW_20S_15DL", "RCW_20S_20DL", "RCW_20S_25DL"]
    
    model_ID = model_ID_list[modelno]
    
    # model_ID    = ["RCW_5S_25DL"][modelno]  
    
    # Import the corresponding wall structure module
    wall_module = importlib.import_module(model_ID)
    wall_module.buildingmodel()
 
    
    file = "Wall Buildings Info.xlsx"
    building_data = pd.read_excel(file, sheet_name=1)
    
    # Find the row in the data corresponding to the model_ID
    row = building_data[building_data.iloc[:, 0] == model_ID]

    # Extract building parameters from the matched row
    NStories = row['No. of stories'].values[0]  # number of stories
    T1 = row['Tcr (sec)'].values[0]             # fundamental period of the structure

    HStory1 = 4.0	                                       # 1st story height in meters
    HStoryTyp = 3.6                                        # story height of other stories in meters
    DT = 1/(500)  /dt_redfac                                       # time step for analysis

    
    # Run the building model with gravity load and damping applied. Set the analysis time to 0.
    # cntrl_nodes = list of nodes at which displacement and acceleration response histories are recorded (from the bottom-most node that is fixed to the topmost node)
    # T1 is the fundamental period of the structure required to read the PSA value corresponding to T1
    # bh is the list of heights of each storey starting from the bottom to the top
    # cntrl_nodes, T1, bh = buildingmodel()
       
    bh = np.zeros(NStories)
    for i in range(NStories):
        if i == 0:
            bh[i] = HStory1
        else:
            bh[i] = HStoryTyp   
    
    def create_cntrlnodes(n):
        return [i for i in range(101, 101 + n)]
    cntrl_nodes = create_cntrlnodes(NStories + 1)
    dri = np.zeros([len(cntrl_nodes)])
    
    ts = np.genfromtxt(gmfolderloc + '/DT_' + str(direction) + '.txt')
    # gacc data
    gacc = np.genfromtxt(gmfolderloc + '/gacc_' + str(gmno) + '_' + str(direction) + '.txt')

    # time step of the GM
    dt = ts[gmno - 1]
    
    gmlen = len(gacc)
    
    # print(model_ID)
    # print(f"The fundamental period for cracked sections is {T1} sec")
    RotD50          = np.genfromtxt(gmfolderloc+'/RotD50.txt')
    psagm         = RotD50[np.argmax(RotD50[:,0]>= T1),gmno]
    scale_factor             = round((SAscaled-Sa_perturb)/psagm*9.81,4)                                                           # Scaling factor as input gms acceleration unit is g's and the models use m as length. This factor also consists scaling factors for IDA

    
    outputdir = "D:\RC_Walls\HC_IDA_Results\IDA output logs"
    sys.stdout    = open(outputdir+'/outputlog_'+str(model_ID)+'_'+str(gmno)+direction+'_'+str(dt_redfac)+'.txt', 'w')
    print ("                                                ")
    print ("################################################")        
    print('fundamental period of the structure: '+str(T1))        
    print('scaled intensity: '+ str(round(scale_factor/9.81*psagm,4))+', scale factor: '+str(scale_factor/9.81)+', unscaled PSA(T1):'+str(round(psagm,3)))
    

    print ("Starting IDA of RC wall-frame model: "+str(model_ID)+" with T1: "+str(T1)+" sec - intensity level - "+str(SAscaled)+"g Ground motion no. - "+str(gmno))
    print("Current value of Sa(T1)= "+str(SAscaled)+" g")
    print('ground motion duration: '+str(gmlen*dt)+' sec')
    
    # time step of the recorded for data
    recdt = 0.01
    
    dir = f"D:\RC_Walls\HC_IDA_Results\{model_ID}\GM" + str(gmno)+ '_' + str(direction) + "/" + str(SAscaled) + 'g'              # Directory
    
    from pathlib import Path
    Path(dir).mkdir(parents=True, exist_ok=True)

    ops.recorder('Node', '-file', dir+"/zTimeHistory_Storey_Displacement.out",  '-dT', recdt, '-time', '-node', *cntrl_nodes, '-dof', 1, 'disp')  

    ################################################################################
    "OUTPUT DATABASE"
    ################################################################################
    opsplt.createODB("Nonlin_RCWall", "RHA/IDA"+str(gmno))
    
    ################################################################################
    # GROUND MOTION INPUT
    ################################################################################
    
    # time series scaled by scale_factor
    ops.timeSeries('Path', 3, '-dt', dt, '-values', *gacc, '-factor', scale_factor)

    ops.recorder('Node', '-file', dir+"/zTimeHistory_Storey_Acc.out", '-timeSeries', 3, '-dT', recdt, '-time', '-node', *cntrl_nodes, '-dof', 1, 'accel')
    ops.pattern('UniformExcitation', 3, 1, '-accel', 3)
    ops.system('UmfPack')
    ops.constraints('Plain')
    ops.numberer('RCM')    
    df = pd.DataFrame(['Analysis running'])
    df.to_csv(dir+'/gmtime'+str(gmno)+'.txt',index=False)
    sys.stderr = open(dir+'/errorlog.txt', 'w')
    maxdri_IDA = SmartAnalyzeTransient(dir, dlim, cntrl_nodes, dri, bh, gmno, DT, int(gmlen*dt/DT))
    print('max drift from IDA:' + str(maxdri_IDA))
    sys.stderr.close()       
    ops.wipeAnalysis()
    ops.wipe()
    return maxdri_IDA


# IDA(gmno, direction, modelno, SAscaled, curdir, curdir_out, foldername, gmfolderloc, dlim, Sa_perturb, dt_redfac)
