# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:30:53 2023

@author: ljp70
"""

""" All units in kN, m, tonnes, and secs"""

from openseespy.opensees import *
import numpy as np
import math
import matplotlib.pyplot as plt
from math import asin, sqrt, pi
import openseespy.postprocessing.Get_Rendering as opsplt
import openseespy.postprocessing.ops_vis as opsv
import openseespy.preprocessing.DiscretizeMember as dm

def buildingmodel():
    wipe()
    
    model('basic', '-ndm', 2, '-ndf', 3)
    ###################################################################################################
    "Define nodes and cordinates"
    ###################################################################################################
    # define structure-geometry parameters
    NStories = 3					               	# number of stories
    NBays = 3			                        	# number of frame bays (excludes bay for P-delta column)
    WBay = 6.0	                                    # bay width in meters
    HStory1 = 4.5	                                  	# 1st story height in meters
    HStoryTyp = 3.6                                   # story height of other stories in meters
    HBuilding = HStory1 + (NStories-1)*HStoryTyp    # height of building
    
    no_of_frames = 4
    
    title = "RC Frame MatTags/03_15matTags.csv"
    
    ###################################################################################################
    "Define section properties and elastic beam column elements"
    ###################################################################################################
    
    "Defining the beam and column properties, area, stiffness...."
    "All dimensions are in kN and m"
            
    E  = 30* 10**6         # kN / m^2
    
    Ic = 0.0052   # m^4       
    Ac = 0.25            # m^2
    
    Ib = 0.0047   # m^4
    Ab = 0.15               # m^2
    
    
    build_height_array = np.zeros(NStories + 1)
    for i in range(NStories + 1):
        build_height_array[i] = HStory1 + HStoryTyp * (i - 1)
        build_height_array[0] = 0
        build_height_array[1] = HStory1
    
    "Saving plots based on type of analyses used"
    
    geomTransf('Corotational', 1)    # Uses the natural deformation scheme, accounting for PDelta and rigid body motion - typically the most accurate
    
    # Element linking matrix
    # Main nodes -  format ij = i:xdirec j: ydirec
    # j:1-5 floors, i: 1,2 main beam column nodes, + 2 centre nodes for the braces
    
    "Main nodes for the structure"
    
    """ Defining the main nodes and coordinates for the model at the ends of each beam and column for the 2D frame
        
        Nodes are defined as follows: numbering = ij where i is the column in the x direction starting at 1,2,3...n
        and j is the row / storey number starting at the ground which is the 1st floor  01, 02, 03...n.
        
        eg. the node at the bottom left is '101' with the coordinates [0.0, 0.0] """
    
    main_coords_x = []
    main_coords_y = []
    for j in range(NStories + 1):
        for i in range(NBays + 1):
            main_coords_x.append(WBay * i)
            main_coords_y.append(build_height_array[j])
    main_coords_y.sort()
    main_coords = np.array([main_coords_x, main_coords_y])   # represents cor1
    
    
    main_nodes = []
    for i in range(1, NStories + 2):
        for j in range(1, NBays + 2):
            if i < 10:
                no = str(j) + str(0) + str(i)
            else:
                no = str(j) + str(i)
            main_nodes.append(int(no))                       # represents nn1
    
    # print(main_nodes)
    # print(main_coords)
    
    "Location of plastic hinges for the beams and columns"
    "Applied a localised plastic hinge assumption"
    
    # Plastic hinges
    # x y a    x = Pier #, y = Floor #, a = location relative to beam-column joint
    # "a" convention: 2 = left; 3 = right;
    # "a" convention: 6 = below; 7 = above;
    
    """ Defining column hinge nodes and their coordinates 
    
        Hinges below the node are denoted '6' while above are denoted '7' 
        The for-loop creates a hinge above and below for every main node in the structure and
        then it is trimmed because the columns at the base will only have a top hinge and 
        likewise for the top column.
        
        """
    column_hinge_nodes = []
    for i in range(6, 8):
        for j in range(len(main_nodes)):
            no = str(main_nodes[j]) + str(i)
            column_hinge_nodes.append(int(no))
    column_hinge_nodes = column_hinge_nodes[NBays + 1 : -(NBays + 1)]
    
    column_hinge_coords_x = np.concatenate((main_coords_x[NBays + 1 :], main_coords_x[: -(NBays + 1)])) 
    column_hinge_coords_y = np.concatenate((main_coords_y[NBays + 1 :], main_coords_y[: -(NBays + 1)])) 
    column_hinge_coords = np.array([column_hinge_coords_x, column_hinge_coords_y])
    
    """ beam hinges: defining the plastic hinge nodes at the ends of each beam + the corresponding coordinates 
    
        The process here works the same. The only difference is that the main_nodes array has been sorted such that
        It counts nodes vertically instead of horizontally. This enables the beam joints to have easily ascribed hinges 
        """
    
    mn = main_nodes[NBays + 1 :]  # removing the base nodes because they don't have beams between them
    mn.sort()
    
    beam_hinge_nodes = []
    beam_hinge_coords = []
    
    for i in range(2, 4):
        for j in range(len(mn)):
            no = str(mn[j]) + str(i)
            beam_hinge_nodes.append(int(no))
            for k in range(len(main_nodes)): 
                if mn[j] == (main_nodes)[k]:
                    beam_hinge_coords.append(main_coords[:, k].tolist())
                    
    beam_hinge_nodes = beam_hinge_nodes[NStories : -NStories]
    beam_hinge_nodes = np.array(beam_hinge_nodes)
    
    beam_hinge_coords = beam_hinge_coords[NStories : -NStories]
    beam_hinge_coords = np.array(beam_hinge_coords).T
    
    
    # Define nodes
    ntag = np.append(np.append(main_nodes, column_hinge_nodes), beam_hinge_nodes)
    cor = np.concatenate((np.concatenate((main_coords, column_hinge_coords), axis=1), beam_hinge_coords), axis=1)
    
    # Nodes
    for i in range(0, len(cor[0, :])):
        node(int(ntag[i]), cor[0, i], cor[1, i])
    
    ###################################################################################################
    "Define boundary conditions"
    ###################################################################################################
    #fixity
    "fix the columns at ground floor - assuming pinned connections for the system (including the leaning column)"
    
    for i in range(NBays + 1):
        fix(main_nodes[i], 1, 1, 1) # fixed
    
    
    "** Note there are multiple nodes at each location, including plastic hinges in columns, beams and the main node"
    
    "m array refers to the massess distributed to each plastic hinge and node within the 'global' node location"
    "ie. frem/12 refers to the mass/3 taken by that node divided by the 4 nodes at that global node"
    
    main_nodes_vert = main_nodes[::]
    main_nodes_vert.sort()
    
    
    ###################################################################################################
    " Define mass distribution of nodes "
    ###################################################################################################
    #Mass distribution
    
    
    "Building self weight derived using NZS1170.5 for a typical office building IL2 Design Life 50 years"
    
    mass_frame = 2590/9.81/3                               # in Tonnes
    # mass_frame = 1
    
    mass_nodes = []
    for i in range(len(main_nodes_vert)):
        if i > NStories and i < len(main_nodes_vert) - NStories - 1:
            x = 2
        else:
            x = 1
        mass_nodes.append(mass_frame * x / (NBays * 2))
        
        if main_nodes_vert[i] % 10 == 1:
            mass_nodes[i] = 0
    
    for i in range(0, len(main_nodes)):
        "mass command is used to set the mass at each node"
        mass(int(main_nodes_vert[i]), mass_nodes[i], mass_nodes[i], 0)
    
    
    
    
    "creating the element tags for the columns"
    # eleID convention:  "1xy" where 1 = col,  x = Pier #, y = Story #
    col_eleTag = main_nodes[: -(NBays+1)]
    for i in range(len(col_eleTag)):
        no = str(1) + str(col_eleTag[i])
        col_eleTag[i] = int(no)
    
    n_col_ele = (NBays + 1) * NStories
    for i in range(0, n_col_ele):
        element('elasticBeamColumn', int(col_eleTag[i]), int(column_hinge_nodes[i]), int(column_hinge_nodes[i + n_col_ele]), float(Ac), E, float(Ic), int(1))
        
    
    "creating the element tags for the beams"
    # eleID convention:  "2xy" where 2 = beam, x = Bay #, y = Floor #
    beam_eleTag = []
    for j in range(1, NBays + 1):
        for i in range(1, NStories + 1):
            no = str(2) + str(j) + str(i)
            beam_eleTag.append(int(no))
            
    n_beam_ele = NBays * NStories
    for i in range(0, n_beam_ele):
        element('elasticBeamColumn', int(beam_eleTag[i]), int(beam_hinge_nodes[i]), int(beam_hinge_nodes[i + n_beam_ele]), float(Ab), E, float(Ib), int(1))
        
        
        
        
    ###################################################################################################
    "Plotting the model"
    ###################################################################################################
    
    # opsv.plot_model(nodes_only=True, axis_off=1)
    # plt.title('Plot of Nodes - including zero-length beam and column hinge nodes')
    
    
    # opsv.plot_model(node_labels=0, axis_off=1)
    # plt.title('Plot of Elements')
    
    ###################################################################################################
    "Define Rotational Springs for Plastic Hinges"
    ###################################################################################################
    
    "Rotational spring (IMK-deterioration model) check notes for details"
    # https://opensees.berkeley.edu/wiki/index.php/Modified_Ibarra-Medina-Krawinkler_Deterioration_Model_with_Bilinear_Hysteretic_Response_(Bilin_Material)
    
    ###################################################################################################
    "Applying the Modified IMK Deterioration Model"
    ###################################################################################################
    
    
    "Common input values for springs - see hyperlink below for more information"
    matTag = np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 0)
    Ky =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 1) 
    My =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 5)
    My_neg =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 6)
    
    for i in range(0, len(matTag)):
        
        pinchx = 1
        pinchy = 1
        damage1 = 0
        damage2 = 0
        beta = 0.6
        θy = My[i] / Ky[i]
        # print(θy)
        Mcr = My[i] / 2
        θcr = 0.01 / 100 * 2
    
        uniaxialMaterial('Hysteretic', int(matTag[i]), 
                          float(Mcr),   float(θcr),  float(My[i]),    float(θy) ,   float(1.1*My[i]),  float(2*θy), 
                          float(-1*Mcr), float(-1*θcr),float(-1*My_neg[i]), float(-1*θy) , float(-1.1*My_neg[i]), float(-2*θy),
                          float(pinchx), float(pinchy), float(damage1), float(damage2), float(beta))
        
        
    
    "material tags 1 - 4 for the rotational capacities of the 'springs' defined above"
    
    # matTag = 1  - Columns 1st floor
    # matTag = 2  - Columns 2nd floor
    # matTag = 3  - Columns 3rd floor
    # matTag = 4  - Beams 
    
    
    "Define column springs"
    # Spring ID: "3xya", where 3 = col spring, x = Pier #, y = Story #, a = location in story
    # "a" convention: 6 = bottom of story, 7 = top of story
    
    column_hinge_nodes
    col_link = main_nodes[NBays + 1 :] + main_nodes[: -(NBays + 1)]
    
    eID_col_hinge = []
    for i in range(0, len(col_link)):
        if i < len(col_link)/2:
            no = str(3) + str(col_link[i]) + str(6)
        else:
            no = str(3) + str(col_link[i]) + str(7)
        eID_col_hinge.append(int(no))
        
        
    # using the material tags from defined rotation capacities based on material properties and dimensions
    matTag = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 
              1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    
    for i in range(0, len(eID_col_hinge)):
        # '-dir', 6 -refers to assigning a spring for rotations about z
        element('zeroLength', int(eID_col_hinge[i]), int(col_link[i]), int(column_hinge_nodes[i]), '-mat', int(matTag[i]), '-dir', 3)
        equalDOF(int(col_link[i]), int(column_hinge_nodes[i]), 1, 2)    # ensures that the spring/hinge displaces the same as the main node
        
    region(1, '-eleRange', min(eID_col_hinge), max(eID_col_hinge))
    
    
    "Define beam springs"
    # Spring ID: "4xya", where 4 = beam spring, x = Pier #, y = Story #, a = location in story
    # "a" convention: 2 = LHS, 3 = RHS
    # refers to springs at the centre of beams on the 2nd and 4th floor as a result of load transfer from bracing
    
    
    beam_link = mn[NStories :] + mn[: -(NStories)]
    beam_hinge_nodes
    
    eID_beam_hinge = []
    for i in range(0, len(beam_link)):
        if i < len(beam_link)/2:
            no = str(4) + str(beam_link[i]) + str(2)
        else:
            no = str(4) + str(beam_link[i]) + str(3)
        eID_beam_hinge.append(int(no))
    
    for i in range(0, len(eID_beam_hinge)):
        # '-dir' , 6 -refers to assigning a spring for rotations about z
        element('zeroLength', int(eID_beam_hinge[i]), int(beam_link[i]), int(beam_hinge_nodes[i]), '-mat', int(4), '-dir', 3)
        equalDOF(int(beam_link[i]), int(beam_hinge_nodes[i]), 1, 2)
        
    region(2, '-eleRange', min(eID_beam_hinge), max(eID_beam_hinge))
    
    
    
    
    ###################################################################################################
    "Applying vertical load to frame to determine the building period"
    ###################################################################################################
    
    "Create the nodal load - command: load nodeID xForce yForce"
    
    # Gravity load
    # opsplt.createODB("Nonlin_RCFrame", "Gravity", Nmodes=3)
    timeSeries('Linear', 1)  # applies the load in a linear manner (not all at once)
    pattern("Plain", 1, 1) # create a plain load pattern - similar to ELF
    
    for i in range(0, len(main_nodes)):
        "mass command is used to set the mass at each node"
        load(int(main_nodes_vert[i]), 0, -mass_nodes[i] * 9.81, 0)
    
    # recorder('Node', '-file', "results/node_disp/NLdisp_pin.out", '-closeOnWrite', '-node', 22, 23, 24, 25, '-dof', 1, 'disp')
    
    
    "(mode shape number, scale, ModelName - Displays the model saved in a database named"
    # opsplt.plot_modeshape(1, 1, Model="Nonlin_RCFrame")
    # 
    # opsplt.plot_model("nodes", "elements")
    
    # create DOF number
    numberer("RCM")
    # create SOE
    system('BandGeneral')
    # create constraint handler
    constraints("Transformation")
    # create number of steps
    nsteps=25
    # create integrator
    integrator('LoadControl', 1/nsteps)
    # create algorithm
    test('RelativeEnergyIncr', 1e-1, 200, 0) # convergence scheme applied to the model
    algorithm("Newton")                      # solution scheme - Newton-Raphson method of solving non-linear equations
    # create analysis object
    analysis("Static")                       # analysis type - ie, static, transient etc..
    # perform the analysis
    # recorder('Node', '-file', "results/modal/eigen.out",'-closeOnWrite','-dof',1,2,3,'eigen')
    # 
    analyze(nsteps)
    
    # opsplt.createODB("Nonlin_RCFrame", "Gravity", Nmodes=3)
    
    # printModel()
    # opsv.plot_model(fig_wi_he=(20., 14.))
    eigen('-genBandArpack', 1)
    record
    loadConst('-time', 0.0)
    wipeAnalysis()
    
    a=eigen(4)
    w1=sqrt(a[0])
    w2=sqrt(a[1])
    
    print(f"T1 : {2*pi/w1:.2f} sec")
    
    zeta=0.02                               # RC frame model damping ratio
    a0    =zeta*2.0*w1*w2/(w1 + w2);  	    # mass damping coefficient based on first and second modes
    a1    =zeta*2.0/(w1 + w2);		        # stiffness damping coefficient based on first and second modes
    rayleigh(a0, 0, 0, a1)
    
    
    def create_cntrlnodes(n):
        return [i for i in range(102, 102 + n - 1)]
    cntrlnodes = create_cntrlnodes(NStories + 1)
    
    ms_1 = np.zeros([len(cntrlnodes)])
    for i in range(0, len(cntrlnodes)):
        ms_1[i] = nodeEigenvector(cntrlnodes[i], 1, 1) # obtains the mode shape of the first mode
    ms_1 = ms_1/(ms_1[-1])                            # normalised the mode shape (sum equal to 1)
    ms_1 = np.round(ms_1, 3)   
    ms_1 = ms_1/sum(ms_1)                            # normalised the mode shape (sum equal to 1)
    
    
    return {
    'cntrlnodes': cntrlnodes,
    'ms_1': ms_1,
    'NBays': NBays,
    'NStories': NStories,
    'HBuilding': HBuilding,
    'mass_frame': mass_frame
    }

buildingmodel()