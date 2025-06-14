# -*- coding: utf-8 -*-
"""
File: SmartAnalyze.py
Created on Sun Jun 28 20:24:24 2020
Author: Difang Huang 黄狄昉

README:
   Introduction
   ----------------------------------------------------------------------------
   The SmartAnalyze provides OpenSees users a easier way to conduct analyses.
   There are two main functions defined in this .py file. SmartAnalyzeTransient & SmartAnalyzeStatic.
   
   SmartAnalyzeTransient is used to conduct time history analyses.
        The arguments must be specified are:
            dt: delta t
            npts: number of points.
   
    SmartAnalyzeStatic is used to conduct static analyze.
        Users provide a loading protocol with displacement targets.
        Then SmartAnalyze will run DisplacementControl analyses accordingly.
        The arguments that must be specified are:
            node: the node tag in the displacement control
            dof: the dof in the displacement control
            maxStep: the maximum step length in the displacement control
            targets: a list of target displacements.
                (E.g. {1 -1 1 -1 0} will result in cyclic load of disp amplitude 1 twice.)
                Note: the first element must be positive.
    
    If the control array is not specified, all the default values will be used.
    If you want to change the control parameters, pass it as an array delegate.
    
    Example
    ---------------------------------------------------------------------------
    Example 1: Basic usage for Transient
        import SmartAnalyze
        constraints('Transformation')
        numberer('Plain')
        system('BandGeneral')
        integrator('Newmark', 0.5, 0.25)
        SmartAnalyzeTransient(dt, npts)
    
    Example 2: Basic usage for Static 
        import SmartAnalyze
        constraints('Transformation')
        numberer('Plain')
        system('BandGeneral')
        protocol=[1, -1, 1, -1, 0]
        SmartAnalyzeStatic(node, dof, maxStep, protocol)
    
    Example 3: change control parameters
        control['printPer']=20
        control['tryAlterAlgoTypes']=True
        control['algoTypes']=[20, 30]
        SmartAnalyzeTransient(dt, npts, control)
        
    Example 4: define user algorithm
        def UserAlgorithm0():
            algorithm('KrylovNewton', increment='initial', maxDim=10)
        control['algoTypes']=[80]
        SmartAnalyzeTransient(dt, npts, control)
    
    The work flow
    ---------------------------------------------------------------------------
        1. Start
        2. Set initial step length, algorithm method and test (You don't need to specify them in your model.)
        3. Divide the whole analysis into pieces. For Static, use maxStep. For Transient, use dt.
        4. Loop by each piece and analyze recursively with RecursiveAnalyze, in the following way
            4.1 Trail analyze for one step, if converge, continue loop 4.
            4.2 If not converge, if tryAddTestTimes is True, if the last test norm is smaller than normTol, recursively set a larger test time.
            4.3 If not converge, if tryAlterAlgoTypes is True, recursively loop to the next algo type.
            4.4 If not converge, divide the current step into two steps. The first one equals to the current step times relaxation.
            4.5 If either step is smaller than minStep:
                4.5.1 If tryLooseTestTol is True, loose test tolerance to looseTestTolTo.
                4.5.2 Else, return not converge code. Exit.
            4.6 If both steps are not smaller than minStep, divide the current piece into two and re-run loop 4.
        5. If converge, return success message.
    
    Control Parameters
    ---------------------------------------------------------------------------
    TEST RELATED:
        `testType`        : string. Identical to the testType in OpenSees test command. Default is "EnergyIncr".
                            Choices see http://opensees.berkeley.edu/wiki/index.php/Test_Command.
        `testTol`         : float. The initial test tolerance set to the OpenSees test command. Default is 1.0e-6.
                            If tryLooseTestTol is set to True, the test tolerance can be loosen.
        `testIterTimes`   : integer. The initial number of test iteration times. Default is 7.
                            If tryAddTestTimes is set to True, the number of test times can be enlarged.
        `testPrintFlag`   : integer. The test print flag in OpenSees Test command. Default is 0.
                            Choices see http://opensees.berkeley.edu/wiki/index.php/Test_Command.
        `tryAddTestTimes` : boolean. Default is True If this is set to True, 
                            the number of test times will be enlarged if the last test norm is smaller than `normTol`,
                            the enlarged number is specified in `testIterTimesMore`.
                            Otherwise, the number of test times will always be equal to `testIterTimes`.
        `normTol`         : float. Only useful when tryAddTestTimes is True. Default is 1.0e3.
                            If unconverge, the last norm of test will be compared to `normTol`.
                            If the norm is smaller, the number of test times will be enlarged.
        `testIterTimesMore` : integer. Only useful when tryaddTestTimes is True. Default is 50.
                            If unconverge and norm is ok, the test iteration times will be set to this number.
        `tryLooseTestTol` : boolean. If this is set to True, if unconverge at minimum step,
                            the test tolerance will be loosen to the number specified by `looseTestTolTo`.
                            the step will be set back.
                            Default is True.
        `looseTestTolTo`  : float. Only useful if tryLooseTestTol is True.
                            If unconvergance at the min step, the test tolerance will be set to this value.
                            Default is 1.0
    
    ALGORITHM RELATED:
        `tryAlterAlgoTypes` : boolean. Default is False.
                              If True, different algorithm types specified in `algoTypes` will be tried during unconvergance.
                              If False, the first algorithm type specified in `algoTypes` will be used.
        `algoTypes`         : list of integer. A list of flags of the algorithms to be used during unconvergance.
                              The integer flag is documented in the following section.
                              Only useful when tryAlterAlgoTypes is True.
                              The first flag will be used by default.
                              The algorithm command in the model will be ignored.
                              Default is { 40 }
                              If you need other algorithm, try a user-defined algorithm. See the following section.
        
    STEP RELATED:
        `initialStep`     : float. Default is equal to $dt.
                            Specifying the initial Step length to conduct analysis.
        `relaxation`      : float, between 0 and 1. Default is 0.5.
                            A factor that is multiplied by each time the step length is shortened.
        `minStep`         : float. Default is 1.0e-6.
                            The step tolerance when shortening the step length.
                            If step length is smaller than minStep, special ways to converge the model will be used according to `try-` flags.
    
    LOGGING RELATED:
        `printPer`        : integer. Print to the console every several trials. Default is 10.
        `debugMode`       : boolean. Print as much information as possible.
    list_algos=[10,11,12,20,21,22,23,30,31,40,41,42,43,44,50,51,52,53,60,70,80]
    Algorithm type flag reference
    ---------------------------------------------------------------------------
     0:  Linear
     1:  Linear -initial
     2:  Linear -factorOnce
    10:  Newton
    11:  Newton -initial
    12:  Newton -initialThenCurrent
    20:  NewtonLineSearch
    21:  NewtonLineSearch -type Bisection
    22:  NewtonLineSearch -type Secant
    23:  NewtonLineSearch -type RegulaFalsi
    30:  ModifiedNewton
    31:  ModifiedNewton -initial
    40:  KrylovNewton
    41:  KrylovNewton -iterate initial
    42:  KrylovNewton -increment initial
    43:  KrylovNewton -iterate initial -increment initial
    44:  KrylovNewton -maxDim 6
    50:  SecantNewton
    51:  SecantNewton -iterate initial
    52:  SecantNewton -increment initial 
    53:  SecantNewton -iterate initial -increment initial
    60:  BFGS
    70:  Broyden    
    80:  PeriodicNewton
    90:  User-defined0
    
    About User-defined algoType:
        If special algorithm is to be used, SmartAyalize provides 3 user-defined algorithms.
        The script author should specify the algorithm as a procedure in the script.
        The script name must be `UserAlgorithm0`, `UserAlgorithm1`, `UserAlgorithm2`.
        Example see section Example No. 4.
        
    Change Log
    ---------------------------------------------------------------------------
    Mon Jun 29 16:10:18 2020 v0.0
        Creat SmartAnalyze.py file.
        
"""

from openseespy.opensees import * 
import time
from math import asin, sqrt
import pandas as pd
import numpy as np
def SmartAnalyzeTransient(dir1, dlim, cnodes, dri, bh, k, dt, npts, ud=''):
    '''
    dt: delta t
    npts: number of points
    ud: change the control parameters in control dict
    '''
    # default control parameters
    control={}
    control['analysis']="Transient"
    control['testType']="NormDispIncr"
    control['testTol']=10**-5
    control['testIterTimes']=25
    control['testPrintFlag']=0
    control['tryAddTestTimes']=True
    control['normTol']=10**3
    control['testIterTimesMore']=50
    control['tryLooseTestTol']=True
    control['looseTestTolTo']=10**-4
    control['tryAlterAlgoTypes']=True
    control['algoTypes']=[4,3,5,1,2]                           
    control['initialStep']=dt
    control['relaxation']=0.5
    control['minStep']=10**-5
    control['printPer']=2500
    control['debugMode']=False
    
    # set user control parameters
    if ud!='':
        userControl=ud                                      
        control.update(userControl)
    
    # print("Control parameters:")
    # for key,value in control.items():
    #     print(key, value)
    
    # initialize analyze commands
    test(control['testType'],control['testTol'],control['testIterTimes'],control['testPrintFlag'])
    setAlgorithm(control['algoTypes'][0])
    analysis('Transient')
    
    # set an array to store current status.
    current={}
    current['startTime']=time.time()
    current['algoIndex']=0
    current['testIterTimes']=control['testIterTimes']
    current['testTol']=control['testTol']
    current['counter']=0
    current['progress']=0
    current['segs']=npts
    
    # divide the whole process into segments.
    #把时程按照数据点分为各个小段seg进行分析
    ab = 0     
    for seg in range(1,npts+1):
        ok=RecursiveAnalyze(control['initialStep'],0,control['testIterTimes'],control['testTol'],control,current)
        #如果递归后不收敛，跳出函数，显示分析失败和用时
       
        for i in range(0,len(dri)-1):        
            disp=nodeDisp(cnodes[i])
            disp1=nodeDisp(cnodes[i+1])
            # globals()["drift_S_"+str(i)].append(abs(disp1[0]-disp[0])/bh[i]*100)
            # dri[i] = max(globals()["drift_S_"+str(i)])
            dri[i]=abs(disp1[0]-disp[0])/bh[i]*100
        
        for i in range(0,len(dri)-1):          
            if dri[i]>dlim:
                ab = 1

        if ab == 1:
                ok=-3
                # print(">>> SmartAnalyze: Completed successfully. Drift > "+str(dlim)+" percent. Time consumption: %f s." %(time.time()-current['startTime']))
                a=[">>> SmartAnalyze: Completed successfully. Drift > "+str(dlim)+" percent. Time consumption: %f s." %(time.time()-current['startTime'])]
                df = pd.DataFrame(a)
                df.to_csv(dir1+'/gmtime'+str(k)+'.txt',index=False)                
                break
        # print('drift at each level is = ',dri)
        # print('a = ',ab)        
        
     
        
       
        if ok<0:
            # print(">>> SmartAnalyze: Analyze failed. Time consumption: %f s." %(time.time()-current['startTime']))
            a=[">>> SmartAnalyze: Analyze failed. Time consumption: %f s." %(time.time()-current['startTime'])]
            df = pd.DataFrame(a)
            df.to_csv(dir1+'/gmtime'+str(k)+'.txt',index=False)                      
            return ok
        
        #该数据点分析收敛，更新成功分析的点数
        current['progress']=seg
        
        #显示实时成功分析的过程占总过程的百分比
        # if control['debugMode']:
            # print("*** SmartAnalyze: progress %f" %(current['progress']/current['segs']))
    
    #全部数据点分析成功，显示分析成功和用时
    if ab!=1:
        # print(">>> SmartAnalyze: Successfully finished! Time consumption: %f s." %(time.time()-current['startTime']))
        a=[">>> SmartAnalyze: Successfully finished! Time consumption: %f s." %(time.time()-current['startTime'])]
        df = pd.DataFrame(a)
        df.to_csv(dir1+'/gmtime'+str(k)+'.txt',index=False)                       
    
    maxdri_IDA = max(dri)
    # print(maxdri_IDA)
    return maxdri_IDA
    

def SmartAnalyzeStatic(HBuilding, stoplim, node, dof, maxStep, targets, ud=''):
    
    
    '''
    node: the node tag in the displacement control
    dof: the dof in the displacement control
    maxStep: the maximum step length in the displacement control
    targets: a list of target displacements, the first element must be positive
    ud: change the control parameters in control dict
    '''
    # set initial step
    if maxStep>targets[0]:
        initialStep=targets[0]
    else:
        initialStep=maxStep
    
    # default control parameters
    control={}
    control['analysis']           = "Static"
    control['testType']           = "NormUnbalance"
    control['testTol']            = 1.0e-3
    control['testIterTimes']      = 5
    control['testPrintFlag']      = 10000
    control['tryAddTestTimes']    = True
    control['normTol']            = 1.0e5
    control['testIterTimesMore']  = 5
    control['tryLooseTestTol']    = True
    control['looseTestTolTo']     = 10**-2.5
    control['tryAlterAlgoTypes']  = True
    control['algoTypes']          = [10,11,20,22,23,25,40,90,91,21,24,41,42,43,44,50,51,52,53,70,71,80,81] 
    control['initialStep']        = initialStep
    control['relaxation']         = 0.05
    control['minStep']            = 1.0e-8
    control['printPer']           = 10000
    control['debugMode']          = False
    
    # set user control parameters
    if ud!='':
        userControl=ud
        control.update(userControl)
    
    # print("Control parameters:")
    # for key,value in control.items():
    #     print(key, value)
    
    # initialize analyze commands
    test(control['testType'],control['testTol'],control['testIterTimes'],control['testPrintFlag'])
    setAlgorithm(control['algoTypes'][0])
    integrator('DisplacementControl', node, dof, initialStep)
    analysis('Static')
    
    # set an array to store current status.
    current={}
    current['startTime']=time.time()
    current['algoIndex']=0
    current['testIterTimes']=control['testIterTimes']
    current['testTol']=control['testTol']
    current['counter']=0
    current['progress']=0
    current['step']=initialStep
    current['node']=node
    current['dof']=dof
    
    # calcuate whole distance; divide the whole process into segments.
    distance=0
    segs=[]                                          #把整个加载过程分解为长度不超过maxStep的，带正负方向的小加载段
    for i in range(len(targets)):                    #目标位移列表循环
        if i==0:                                     #第一段位移
            section=targets[0]                       #推覆位移直接就是列表第一个位移
            positive=True                            #正向 
        else:                                        #非第一段位移
            section=targets[i]-targets[i-1]          #推覆位移为目标位移减去上一段目标位移
            if section>=0:                           #若大于零为正向推
                positive=True           
            else:                                    #若小于零为负向推
                positive=False
        
        distance=distance+abs(section)               #所推的绝对位移总和，section为每次单推的距离
        
        if positive:                                 #若目前处于正向加载
            j=0
            while (section-j*maxStep)>maxStep:       #把本段推覆位移分为长度为maxStep的各个小段，存于列表segs中
                segs.append(maxStep)
                j+=1
            segs.append(section-j*maxStep)           #最后一小段不足maxStep的也存于segs
            
        else:                                        #若目前处于负向加载
            j=0
            while (-section-j*maxStep)>maxStep:      #把本段推覆位移分为长度为maxStep的各个小段，存于列表segs中
                segs.append(-maxStep)
                j+=1
            segs.append(section+j*maxStep)           #最后一小段不足maxStep的也存于segs
    
    current['segs']=len(segs)                        #整个加载过程中所有小加载段的个数
    # print(segs)
    # print('')
    # print(sum(segs))
    # print('')
    # Run recursive analysis
    #对每个小加载段进行计算

    ok = 0
    tar_dri  = stoplim
    tar_disp = round(stoplim*HBuilding,4)
    maxU     = int(targets[0]/maxStep*100)
    currentDisp = 0
    baseshear = np.zeros(maxU)
    top_drift = np.zeros(maxU)
    top_disp = np.zeros(maxU)
    icount = 0
    unload="False"
    for seg in segs:
        # ok=RecursiveAnalyze(seg, 0, control['testIterTimes'], control['testTol'], control, current)
    
    #########################################################################################################
        

        # print("--------------------------------------------------------------------------------")
    
        
        # while ok == 0 and currentDisp <= tar_disp+0.0001*tar_disp:
            
        ok=RecursiveAnalyze(seg, 0, control['testIterTimes'], control['testTol'], control, current)
        
    #########################################################################################################

        SmartAnalyzeStatic.aprog=ok
        if ok<0:               #若不收敛，跳出函数并显示用时
            print(">>> SmartAnalyze: Analyze failed. Time consumption: %f s." %(time.time()-current['startTime']))
            
            return ok
        
        # if nodeDisp(node, 1)/15.3*100 >1 and getLoadFactor(2)<perred:
        #     print("Analysis terminated")
        #     break         
        
        #收敛，成功分析的过程数+1
        current['progress']+=1
        
        #显示成功分析的过程数占总的加载段数的百分比
        # if control['debugMode']:
        #     print("*** SmartAnalyze: progress %f" %(current['progress']/current['segs']))
    
    #全部加载段分析完成，显示成功完成分析和用时
    # if ok != 0 or tar_disp-0.0001*tar_disp<currentDisp < tar_disp+0.0001*tar_disp or (unload=="True" and baseshear[i-1]<perred*peakbs):
    #     if ok != 0:
    #           print("Algortihm failed to converge")
    #     elif (unload=="True" and baseshear[icount-1]<perred*peakbs):
    #               print("Pushover analysis has been terminated as current base shear is less than " + str( perred*100)+ "% of the peak value")
    #     elif (tar_disp-0.0001*tar_disp<currentDisp < tar_disp+0.0001*tar_disp):
    #               print("Pushover analysis has been terminated as current top storey displacement is equal to the target displacement of " + str( tar_disp)+ "m")

    print(">>> SmartAnalyze: Successfully Finished! Time consumption: %f s." %(time.time()-current['startTime']))
    
    
    
    
    

def RecursiveAnalyze(step, algoIndex, testIterTimes, testTol, vcontrol, vcurrent):
    '''
    step: 步长，动力分析为dt; 静力分析为小加载段的位移，<=maxStep
    algoIndex: 初始迭代方法列表的序号，一般从第一个开始，即为0
    testIterTimes: 迭代次数，默认为7
    testTol: 迭代容差，默认为1.0e-6
    vcontrol: 控制参数字典
    vcurrent: 状态参数字典
    '''
    control=vcontrol
    current=vcurrent
    
    # print('CONTROL PARAMETERS:')
    # print(control)
    # print('CURRENT STATE PARAMETERS:')
    # print(current)
    # print('\n')
    
    #输出本次分析的参数
    # if control['debugMode']:
    #     print("*** SmartAnalyze: Run Recursive: step=%f, algoI=%i, times=%i, tol=%f" %(step, algoIndex, testIterTimes, testTol))
    #     print('\n')
    
    #改变迭代方法
    if algoIndex!=current['algoIndex']:
        # print(">>> SmartAnalyze: Setting algorithm to %i" %(control['algoTypes'][algoIndex]))
        # print('\n')
        setAlgorithm(control['algoTypes'][algoIndex])
        current['algoIndex']=algoIndex
    
    #变化迭代次数和容差
    if testIterTimes!=current['testIterTimes'] or testTol!=current['testTol']:
        if testIterTimes!=current['testIterTimes']:
            # print(">>> SmartAnalyze: Setting test iteration times to %i" %(testIterTimes))
            # print('\n')
            current['testIterTimes']=testIterTimes
        if testTol!=current['testTol']:
            # print("SmartAnalyze: Setting test tolerance to %f" %(testTol))
            # print('\n')
            current['testTol']=testTol
            
        test(control['testType'], testTol, testIterTimes, control['testPrintFlag'])
    
    #静力分析修改步长
    if control['analysis']=='Static' and current['step']!=step:
        # print(">>> SmartAnalyze: Setting step to %f" %(step))
        # print('\n')
        integrator('DisplacementControl', current['node'], current['dof'], step)
        current['step']=step
    
    # trial analyze once
    #静力分析
    if control['analysis']=='Static':
        ok=analyze(1)
        
    #动力分析
    else:
        ok=analyze(1, step)  

    #单次分析次数记录
    current['counter']+=1
    
    #分析收敛 跳出函数 开始下一小段的计算
    if ok==0:
        #满指定次数就输出完成百分比和用时
        if current['counter']>=control['printPer']:
            # print("* SmartAnalyze: progress %f. Time consumption: %f s." 
            #       %(current['progress']/current['segs'], (time.time()-current['startTime'])/1000.0))
            # print('\n')
            current['counter']=0
        return 0
    
    # 分析不收敛 开始改变参数递归
    # Add test iteration times. Use current step, algorithm and test tolerance.
    if control['tryAddTestTimes'] and testIterTimes!=control['testIterTimesMore']:
        #检查范数
        norm=testNorm()
        #如果范数小于规定值，增加迭代次数，其他参数不变，本函数递归
        if norm[-1]<control['normTol']:
            # print(">>> SmartAnalyze: Adding test times to %i." %(control['testIterTimesMore']))
            # print('\n')
            return RecursiveAnalyze(step, algoIndex, control['testIterTimesMore'], testTol, control, current)
        #如果范数大于规定值，不增加
        # else:
        #     # print(">>> SmartAnalyze: Not adding test times for norm %f" %(norm[-1]))
        #     # print('\n')
    
    # Change algorithm. Set back test iteration times.
    # 如果迭代方法序号还没超 尝试下一种迭代方法
    if control['tryAlterAlgoTypes'] and (algoIndex+1)<len(control['algoTypes']):
        algoIndex+=1
        # print(">>> SmartAnalyze: Setting algorithm to  %i." %(control['algoTypes'][algoIndex]))
        # print('\n')
        return RecursiveAnalyze(step, algoIndex, testIterTimes, testTol, control, current)
    
    # If step length is too small, try add test tolerance. set algorithm and test iteration times back.
    if abs(step)<2*control['minStep']:
        # print(">>> SmartAnalyze: current step %f is too small!" %(step))
        # print('\n')
        if control['tryLooseTestTol'] and current['testTol']!=control['looseTestTolTo']:
            # print("!!! SmartAnalyze: Warning: Loosing test tolerance")
            # print('\n')
            return RecursiveAnalyze(step, 0, control['testIterTimes'], control['looseTestTolTo'], control, current)
        
        # Here, all methods have been tried. Return negative value.
        return -1
    
    # Split the current step into two steps.
    stepNew=step*control['relaxation']
    if stepNew>0 and stepNew<control['minStep']:
        stepNew=control['minStep']
    
    if stepNew<0 and stepNew>-control['minStep']:
        stepNew=-control['minStep']
    
    stepRest=step-stepNew
    # print(">>> SmartAnalyze: Dividing the current step %f into %f and %f" %(step, stepNew, stepRest))
    # print('\n')
    ok=RecursiveAnalyze(stepNew, 0, testIterTimes, testTol, control, current)
    if ok<0:
        return -1
    ok=RecursiveAnalyze(stepRest, 0, testIterTimes, testTol, control, current)
    if ok<0:
        return -1
    return 1

def setAlgorithm(algotype):
    
        def case0():

            algorithm("Linear")


        def case1():
    
            algorithm("KrylovNewton")
            
        def case2():
    
            algorithm("Broyden",15)            
            
        def case3():
    
            algorithm("NewtonLineSearch", "-type", "Secant")            
    
        def case4():
    
            algorithm("Newton")
    
        def case5():
    
            algorithm("PeriodicNewton")
            
        def case6():

            algorithm("ExpressNewton")

        def case100():
            # User algorithm0
            if user_algo_args is not None:
                algorithm(*user_algo_args)

        def default():
              print("!!! SmartAnalyze: ERROR! WRONG Algorithm Type!")

        switch = {
            0: case0,
            1: case1,
            2: case2,
            3: case3,
            4: case4,
            5: case5,
            6: case6,
            }
        
        # choice=str(algotype)
        # switch.get(choice, default)()
        switch.get(algotype, default)()    


























