'Improting the required packages'
from mpi4py import MPI
from mpi_master_slave import Master, Slave
from mpi_master_slave import WorkQueue
import time
import numpy as np
import os
from pathlib import Path
import pandas as pd
from IDA_parallel_Pledger import IDA

'Inputs'
direcs          = ['x', 'y']                               # Gm direction. strong axis of the gravity system oriented along 'x' and the weak axis along 'y'
delta_SaT1      = 0.05                                      # intensity increment
maxSaT1         = 15                                       # maximum SaT1 in gs
SAscale         = np.arange(0.1,maxSaT1,delta_SaT1)        # array of SaT1 values for scaling
SAscale         = np.round(SAscale,2)                      # rounding all the values off to 2 decimal places
gms             = 50                                      # total number of ground motions for each intensity level
dlim            = 5                                        # drift limit (%) to terminate analysis the analysis. This is representative of structural collapse
ntrials         = 0                                         # ntrials+1 is the number of times bisection is performed to obtain a precise estimate of the collapse intensity                         
Sa_perturb      = 0
dt_redfac       = 1
n_models        = 4

foldername      = 'HC_IDA_Results'                            # folder name to save the results 
gmfolderloc     = 'D:\HC_IDA_GMs'                            # location of the ground motion set

'required function files to run the analysis'
def directories(mno,k,SAscaled,foldername):
    curdir     = 'D:\RC_Walls' # current directory of the code location
    
    # models      = ["RCW_5S_10DL" , "RCW_5S_15DL" , "RCW_5S_20DL" , "RCW_5S_25DL",
    #                 "RCW_10S_10DL", "RCW_10S_15DL", "RCW_10S_20DL", "RCW_10S_25DL",
    #                 "RCW_15S_10DL", "RCW_15S_15DL", "RCW_15S_20DL", "RCW_15S_25DL",
    #                 "RCW_20S_10DL", "RCW_20S_15DL", "RCW_20S_20DL", "RCW_20S_25DL"] 
    
    models      = ["RCW_20S_10DL", "RCW_20S_15DL", "RCW_20S_20DL", "RCW_20S_25DL"]
    
    # models      = ["RCW_5S_25DL"]  
    
    # selecting the model parameters on the specific model chosen
    model      = models[mno]
    "-------------------------------- analysis results directory --------------------------------------"         
    curdir_out = curdir
    outputdir     = curdir_out+"/"+foldername+"/"+model
    return outputdir,curdir,curdir_out,foldername
    
    

def failure_troubleshoot(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,dt_redfac):
    'if the analysis fails, intensity is reduced by 0.01. The analysis time step is further reduced by a factor of 2 if it fails again'
    Sa_perturb = 0; dt_redfac = 1.5
    maxdri_IDA = IDA(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,Sa_perturb,dt_redfac)
    if maxdri_IDA < 0:
        Sa_perturb = 0;dt_redfac = 2
        maxdri_IDA = IDA(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,Sa_perturb,dt_redfac)
        if maxdri_IDA < 0:
            Sa_perturb = 0.01; dt_redfac = 1.5
            maxdri_IDA = IDA(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,Sa_perturb,dt_redfac)
            if maxdri_IDA < 0:
                Sa_perturb = -0.01; dt_redfac = 1.5
                maxdri_IDA = IDA(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,Sa_perturb,dt_redfac)
    
    return maxdri_IDA



def main():  
    name = MPI.Get_processor_name()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    print('current processor %d (total processors %d)' % ( rank, size) )
    print(f'Processor {rank} of {size} on {name} is starting.')
    
    if rank == 0: # Master
        begin = time.time()
        app = MyApp(slaves = range(1, size))
        app.run(n_models, gms, direcs, dlim, Sa_perturb,dt_redfac,gmfolderloc,foldername)
        app.terminate_slaves()
        end = time.time()
        print(f"Total runtime of the program is {(end - begin)/60/60} hours")
    else: # Any slave
        MySlave().run()
    
    print('Task completed (rank %d)' % (rank) )


    



        
        
class MyApp(object):
    """
    This is my application that has a lot of work to do so it gives work to do
    to its slaves until all the work is done
    """
    def __init__(self, slaves):
        # when creating the Master we tell it what slaves it can handle
        self.master = Master(slaves)
        # WorkQueue is a convenient class that run slaves on a tasks queue
        self.work_queue = WorkQueue(self.master)
    def terminate_slaves(self):
        """
        Call this to make all slaves exit their run loop
        """
        self.master.terminate_slaves()
    def run(self, n_models, gms, direcs, dlim, Sa_perturb,dt_redfac,gmfolderloc,foldername):
        """
        This is the core of my application, keep starting slaves
        as long as there is work to do
        """
        #
        # let's prepare our work queue. This can be built at initialization time
        # but it can also be added later as more work become available
        for mno in range(0, n_models):
            for z in range(0,len(SAscale)):
                for mdir in direcs:
                    for k in range (1,gms+1):
                        self.work_queue.add_work(data = (k,SAscale[z],mdir,mno,delta_SaT1,dlim,ntrials, Sa_perturb,dt_redfac,foldername,gmfolderloc))       
        
        # Keeep starting slaves as long as there is work to do
        while not self.work_queue.done():
            # give more work to do to each idle slave (if any)
            self.work_queue.do_work()
            # reclaim returned data from completed slaves
            for slave_return_data in self.work_queue.get_completed_work():
                done, message, rankno = slave_return_data
                if done:                    
                    print('Processor "%d" - "%s"' % ( rankno, message) )

            # sleep some time
            time.sleep(0)

class MySlave(Slave):
    """
    A slave process extends Slave class, overrides the 'do_work' method
    and calls 'Slave.run'. The Master will do the rest
    """
    def __init__(self):
        super(MySlave, self).__init__()

    def do_work(self, data):
        rank = MPI.COMM_WORLD.Get_rank()
        name = MPI.Get_processor_name()
        gmno, SAscaled, direction, modelno,delta_Sa,dlim,ntrials,Sa_perturb,dt_redfac,foldername,gmfolderloc = data  # read data from the master
        outputdir,curdir,curdir_out,foldername = directories(modelno,gmno,SAscaled,foldername)             # read the directory locations
        outputdir = outputdir+'/GM'+str(gmno) + '_' + str(direction)                                # set the directory for a specific model subjected to a specific gm and a specific intensity
        
        # print("GM no. = " + str(gmno))
        # print(f"Output directory = {outputdir}")
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)                                                   # create directory if it doesnt exist  
        file_list = os.listdir(outputdir)                                            # get file list of intensities udner a specific gm   
        
        # print("The file list at directory (previously run intensities : " + str(file_list))
        for zdummy in range(0,len(file_list)):                                       # check the analysis output if any previous analysis exists  
            # print("GM no. = " + str(gmno))
            gmtime   = outputdir+'/'+file_list[zdummy]+'/gmtime'+str(gmno)+'.txt'
            # print("gmtime directory : " + gmtime)
            if Path(gmtime).is_file():
                colcount = pd.read_csv(gmtime) 
                check_crit = 'Drift > '+str(dlim)
                if check_crit in str(colcount.iloc[0][0]):                 # set collapse flag as true if the strucutre has collapsed  
                    collapse = 'true'
                    break
                else:
                   collapse = 'false'                 # set collapse flag as false if the strucutre has not collapsed  
        if len(os.listdir(outputdir)) == 0:           # set collapse flag as false if no analysis files have been found. (no previous analysis has been run) 
               collapse = 'false'                     
        # print("The building has collapse?  " + collapse)       
        if  collapse != 'true':                 # run a specific model with a specific gmno and intensity if collapse is not true                                                   
            maxdri_IDA = IDA(gmno, direction, modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,Sa_perturb,dt_redfac)
            # print(f"Maximum storey drift = {maxdri_IDA:.2f} %")
            if maxdri_IDA < 0:
                maxdri_IDA = failure_troubleshoot(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,dt_redfac)
            # In case of failure the code below is activated 
            'Bisection algorithm is performed for ntrials+1 number of times to estimate the collapse intensity with finer precision'

            # failcrit = dlim; demandval = maxdri_IDA
            # if demandval > failcrit:                 
            #     delx = 2
            #     SAscaled   = np.round(SAscaled - delta_Sa/delx ,4) ; delx         = delx*2
            #     maxdri_IDA = IDA(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,Sa_perturb,dt_redfac)
            #     if maxdri_IDA < 0:
            #         maxdri_IDA = failure_troubleshoot(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,dt_redfac)
            #     for trial in range(0,ntrials):
            #         failcrit = dlim; demandval = maxdri_IDA
            #         if demandval > failcrit:
            #             SAscaled   = np.round(SAscaled - delta_Sa/delx ,4);delx         = delx*2
            #             maxdri_IDA = IDA(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,Sa_perturb,dt_redfac)
            #             if maxdri_IDA < 0:
            #                 maxdri_IDA = failure_troubleshoot(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,dt_redfac)
            #             SAscaled   = np.round(SAscaled + delta_Sa/delx ,4);delx         = delx*2   
            #             maxdri_IDA = IDA(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,Sa_perturb,dt_redfac)
            #             if maxdri_IDA < 0:
            #                 maxdri_IDA = failure_troubleshoot(gmno,direction,modelno,SAscaled,curdir,curdir_out,foldername,gmfolderloc,dlim,dt_redfac)   
        return (True, 'model no: '+str(modelno)+' Completed ground motion '+str(gmno)+ str(direction)+' at level '+str(round(SAscaled,2))+'g',rank)


if __name__ == "__main__":
    main()
    

    
    
