# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:19:48 2024

@author: ljp70
"""

import openseespy.opensees as ops
import numpy as np
import pandas as pd
import openseespy.postprocessing.Get_Rendering as opsplt

def buildingmodel():
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    ops.geomTransf('Corotational', 1)    # Uses the natural deformation scheme, accounting for PDelta and rigid body motion - typically the most accurate
    
    title = "RCW_10S_10DL_matTags.csv"
    file = "Wall Buildings Info.xlsx"
    building_data = pd.read_excel(file, sheet_name=1)
    
    # all units in kN, m, and sec
    """ Extract all of these values from an array from excel 
     i.e. If building ID  = 1, use values in first row."""
    
    ID = 8
    
    # Wall dimensions
    tw = building_data.loc[ID-1, 'tw'] /1000
    lw = building_data.loc[ID-1, 'lw'] /1000
    
    # column dimensions
    hc = building_data.loc[ID-1, 'hc'] /1000
    
    # beam properties
    db = building_data.loc[ID-1, 'db'] /1000
    wb = building_data.loc[ID-1, 'wb'] /1000
    
    # concrete elastic modulus
    Ec  = 25* 10**6           # kN / m^2
    
    # define structure-geometry parameters
    NStories = building_data.loc[ID-1, 'No. of stories']	        # number of stories
    NBays =  building_data.loc[ID-1, 'No. of bays']	        # number of frame bays (excludes bay for P-delta column)
    
    
    if building_data.loc[ID-1, 'Wall type']	== 'R':
        # Define wall section properties
        m = 10  # no of sections to split the wall into.
        # Thickness of each element
        thick = [tw] * m          
        # Width of each element in mm
        widths = [lw/m] * m       
        # Reinforcement ratio of each element
        rho = [0.03, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.03]        
    
    else: 
        # Define wall section properties
        m = 10  # no of sections to split the wall into.
        # Thickness of each element
        thick = [hc, hc, tw, tw, tw, tw, tw, tw, hc, hc]           
        # Width of each element in mm
        r_len = lw - 2*hc
        
        widths = [hc/2, hc/2, r_len/6, r_len/6, r_len/6, r_len/6, r_len/6, r_len/6, hc/2, hc/2]       
        # Reinforcement ratio of each element
        rho = [0.03, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.03]   
    
    
    Ic = hc**4/12 *1.1 / 3       # m^4       
    Ac = hc**2 *2                  # m^2
    
    Ib = 1.5* db**3 * wb/12 *1.1 / 3  # m^4
    Ab = db*wb *3                # m^2
    
    WBay = 6.0	                                        # bay width in meters
    HStory1 = 4.0	                                   # 1st story height in meters
    HStoryTyp = 3.6                                   # story height of other stories in meters
    HBuilding = HStory1 + (NStories-1)*HStoryTyp      # height of building
    
    mass_frame = 8 * 6**2 * (NBays+ 1) / 9.81                    # in Tonnes
    
    build_height_array = np.zeros(NStories + 1)
    for i in range(NStories + 1):
        build_height_array[i] = HStory1 + HStoryTyp * (i - 1)
        build_height_array[0] = 0
        build_height_array[1] = HStory1
    
    
    main_coords_x = []
    main_coords_y = []
    for j in range(NStories + 1):
        for i in range(NBays):
            main_coords_x.append(WBay * (i+1))
            main_coords_y.append(build_height_array[j])
    main_coords_y.sort()
    # print(main_coords_x)
    main_coords = np.array([main_coords_x, main_coords_y])   # represents cor1
    
    
    # Initialising main_nodes for columns. Assuming wall is at the start of the frame. Therefore columns start at vertical column 2 (j = 2)
    main_nodes = []
    for i in range(1, NStories + 2):
        for j in range(2, NBays + 2):
            if i < 10:
                no = str(j) + str(0) + str(i)
            else:
                no = str(j) + str(i)
            main_nodes.append(int(no))                       # represents nn1
            
            
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
    column_hinge_nodes = column_hinge_nodes[NBays : -(NBays)]
    
    column_hinge_coords_x = np.concatenate((main_coords_x[NBays:], main_coords_x[: -(NBays)])) 
    column_hinge_coords_y = np.concatenate((main_coords_y[NBays:], main_coords_y[: -(NBays)])) 
    column_hinge_coords = np.array([column_hinge_coords_x, column_hinge_coords_y])
    
    """ beam hinges: defining the plastic hinge nodes at the ends of each beam + the corresponding coordinates 
    
        The process here works the same. The only difference is that the main_nodes array has been sorted such that
        It counts nodes vertically instead of horizontally. This enables the beam joints to have easily ascribed hinges 
        """
    
    # Updating main_nodes and main_coords array to include wall nodes (j = 1). Beams connect to main nodes of wall. 
    main_nodes = []
    for i in range(1, NStories + 2):
        for j in range(1, NBays + 2):
            if i < 10:
                no = str(j) + str(0) + str(i)
            else:
                no = str(j) + str(i)
            main_nodes.append(int(no))                       # represents nn1
            
    main_coords_x = []
    main_coords_y = []
    for j in range(NStories + 1):
        for i in range(NBays + 1):
            main_coords_x.append(WBay * (i))
            main_coords_y.append(build_height_array[j])
    main_coords_y.sort()
    # print(main_coords_x)
    main_coords = np.array([main_coords_x, main_coords_y])   # represents cor1
    
    
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
    
    # Define element nodes
    Wall_nodes = []
    for i in range(1, NStories + 1):
        if i < 10:
            no1 = str(10) + str(i)
            no2 = int(no1) + 1
        else:
            no1 = str(1) + str(i)
            no2 = int(no1) + 1
        Wall_nodes.append([int(no1), int(no2)])
    
    # Define nodes
    ntag = np.append(np.append(main_nodes, column_hinge_nodes), beam_hinge_nodes)
    cor = np.concatenate((np.concatenate((main_coords, column_hinge_coords), axis=1), beam_hinge_coords), axis=1)
    
    # Nodes
    for i in range(0, len(cor[0, :])):
        ops.node(int(ntag[i]), cor[0, i], cor[1, i])
    
    # Define material properties for ConcreteCM confined 
    matTag_concrete = 100            # confined concrete
    fpcc = 51*10**3                # Peak compressive strength kN/m2
    epcc = -0.011                  # Strain @ peak compressive strength
    # fpcc and epcc based on Mander J.B., Priestley M.J.N., and Park R. (1988). “Theoretical Stress-Strain Model for Confined Concrete”, ASCE Journal of Structural Engineering, V. 114, No. 8, pp. 1804-1826.
    rc = fpcc/1000/6.68 - 1.85    # Ratio of unloading slope to initial slope
    # rc based on : Tsai W.T. (1988), “Uniaxial Compressional Stress-Strain Relation of Concrete”, ASCE Journal of Structural Engineering, V. 114, No. 9, pp. 2133-2136.
    xcrn = 1.03                 # Compressive strain at which reloading begins
    ft = 2.8*10**3              # Tensile strength
    et = 0.00008                # Ultimate tensile strain
    rt = 1.5                    # Tension softening stiffness ratio
    xcrp = 0.000001             # Tensile strain at which tension stiffening begins
    gap_close = 1               # Gap closing parameter
    
    
    # Define material properties
    ops.uniaxialMaterial('ConcreteCM', matTag_concrete, -fpcc, epcc, Ec, rc, xcrn, ft, et, rt, xcrp, '-GapClose', gap_close)
    
    # Define material properties for SteelMPF
    matTag_steel = 101
    fyp = 540*10**3             # Yield strength in tension
    fyn = 540*10**3            # Yield strength in compression
    E0_steel = 200*10**6        # Initial elastic modulus of steel
    bp = 0.01                   # Strain-hardening ratio in tension
    bn = 0.01                  # Strain-hardening ratio in compression
    a1 = 0.0
    a2 = 1.0
    a3 = 0.0
    a4 = 1.0
    
    R0=20; cR1=0.925; cR2=0.15
    params=[R0,cR1,cR2]
    
    ops.uniaxialMaterial('SteelMPF', matTag_steel, fyp, fyn, E0_steel, bp, bn, *params, a1, a2, a3, a4)
    
    # SHEAR ........................................................
    # uniaxialMaterial Elastic $matTag $E <$eta> <$Eneg>
    # NOTE: large shear stiffness assigned since only flexural response
    Ag = tw * lw  				   	# Gross area of the wall cross section
    mu = 0.1
    G=Ec/(2*(1+mu))					# Shear Modulus
    shear_matTag = 102
    
    # Build shear material
    ops.uniaxialMaterial('Elastic', shear_matTag, G*Ag)
    
    concrete_list = [matTag_concrete] * m
    steel_list = [matTag_steel] * m
    
    
    density = 2350*9.81/1000  # density of the wall in units kN/m^3
    density = 0
    c = 0.4 # location of the center of rotation
    
    
    # Define MVLEM elements
    for i in range(NStories):
        eleTag = i + 1
        ops.element('MVLEM',int(eleTag), float(density), *Wall_nodes[i], int(m), float(c), '-thick', *thick, '-width', *widths, '-rho', *rho, '-matConcrete', *concrete_list,'-matSteel', *steel_list, '-matShear', shear_matTag)
    
    
    for i in range(NBays + 1):
        ops.fix(main_nodes[i], 1, 1, 1) # fixed
    
    
    ###################################################################################################
    " Define mass distribution of nodes "
    ###################################################################################################
    #Mass distribution
    
    main_nodes_vert = main_nodes[::]
    main_nodes_vert.sort()
    
    "Building self weight derived using NZS1170.5 for a typical office building IL2 Design Life 50 years"
    
    
    
    mass_nodes = []
    for i in range(len(main_nodes_vert)):
        mass_nodes.append(mass_frame / (NBays + 1))
        
        if main_nodes_vert[i] % 100 == 1:
            mass_nodes[i] = 0
    
    for i in range(0, len(main_nodes)):
        "mass command is used to set the mass at each node"
        ops.mass(int(main_nodes_vert[i]), mass_nodes[i], mass_nodes[i], 0)
    
    
    # Initialising main_nodes for columns. Assuming wall is at the start of the frame. Therefore columns start at vertical column 2 (j = 2)
    main_nodes = []
    for i in range(1, NStories + 2):
        for j in range(2, NBays + 2):
            if i < 10:
                no = str(j) + str(0) + str(i)
            else:
                no = str(j) + str(i)
            main_nodes.append(int(no))                       # represents nn1
    
    "creating the element tags for the columns"
    # eleID convention:  "1xy" where 1 = col,  x = Pier #, y = Story #
    col_eleTag = main_nodes[: -(NBays)]
    for i in range(len(col_eleTag)):
        no = str(1) + str(col_eleTag[i])
        col_eleTag[i] = int(no)
    
    n_col_ele = (NBays) * NStories
    for i in range(0, n_col_ele):
        ops.element('elasticBeamColumn', int(col_eleTag[i]), int(column_hinge_nodes[i]), int(column_hinge_nodes[i + n_col_ele]), float(Ac), Ec, float(Ic), int(1))
        
    
    "creating the element tags for the beams"
    # eleID convention:  "2xy" where 2 = beam, x = Bay #, y = Floor #
    beam_eleTag = []
    for j in range(1, NBays + 1):
        for i in range(1, NStories + 1):
            no = str(2) + str(j) + str(i)
            beam_eleTag.append(int(no))
            
    n_beam_ele = NBays * NStories
    for i in range(0, n_beam_ele):
        ops.element('elasticBeamColumn', int(beam_eleTag[i]), int(beam_hinge_nodes[i]), int(beam_hinge_nodes[i + n_beam_ele]), float(Ab), Ec, float(Ib), int(1))
        
    
        
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
    # https://opensees.berkeley.edu/wiki/index.php/Modified_Ibarra-Medina-Krawinkler_Deterioration_Model_with_Bilinear_Hysteretic_Response_(Bilin_Material)
    
    matTag = np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 0)
    Ky =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 1) *11
    θp =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 2)
    θpc =    np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 3)
    th_uP =  np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 4)
    My =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 5)
    My_neg = np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 6)
    McMy =   np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 7)
    ResP =   np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 8)
    LS =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 9)   
    LC =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 10)  
    LA =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 11)  
    LK =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 12)  
    cS =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 13)
    cC =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 14)
    cA =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 15)
    cK =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 16)
    DP =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 17)
    DN =     np.loadtxt(title, delimiter = ",", skiprows=1, usecols= 18)
    
    for i in range(0, len(matTag)):
        ops.uniaxialMaterial('IMKPeakOriented', int(matTag[i]), float(Ky[i]), float(θp[i]), float(θpc[i]), float(th_uP[i]), float(My[i]), float(McMy[i]), 
                         float(ResP[i]), float(θp[i]), float(θpc[i]), float(th_uP[i]), float(My_neg[i]), float(McMy[i]), float(ResP[i]),
                         float(LS[i]), float(LC[i]), float(LA[i]), float(LK[i]), float(cS[i]), float(cC[i]), float(cA[i]), 
                         float(cK[i]), float(DP[i]), float(DN[i]))
    
    "material tags 1 - 6 for the rotational capacities of the 'springs' defined above"
    
    # matTag = 1  - Columns 1st floor
    # matTag = 2  - Columns 2nd floor
    # matTag = 3  - Columns 3rd floor.....
    # matTag = 4  - Beams 
    
    "Define column springs"
    # Spring ID: "3xya", where 3 = col spring, x = Pier #, y = Story #, a = location in story
    # "a" convention: 6 = bottom of story, 7 = top of story
    
    column_hinge_nodes
    col_link = main_nodes[NBays :] + main_nodes[: -(NBays)]
    
    eID_col_hinge = []
    for i in range(0, len(col_link)):
        if i < len(col_link)/2:
            no = str(3) + str(col_link[i]) + str(6)
        else:
            no = str(3) + str(col_link[i]) + str(7)
        eID_col_hinge.append(int(no))
        
        
    # using the material tags from defined rotation capacities based on material properties and dimensions
    matTag = []
    for k in range(0, 2):
        for i in range(1, NStories + 1):
            for j in range(0, NBays + 1):
                matTag.append(i)
    
    for i in range(0, len(eID_col_hinge)):
        # '-dir', 6 -refers to assigning a spring for rotations about z
        ops.element('zeroLength', int(eID_col_hinge[i]), int(col_link[i]), int(column_hinge_nodes[i]), '-mat', int(matTag[i]), '-dir', 6)
        ops.equalDOF(int(col_link[i]), int(column_hinge_nodes[i]), 1, 2)    # ensures that the spring/hinge displaces the same as the main node
        
    ops.region(1, '-eleRange', min(eID_col_hinge), max(eID_col_hinge))
    
    
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
        ops.element('zeroLength', int(eID_beam_hinge[i]), int(beam_link[i]), int(beam_hinge_nodes[i]), '-mat', int(NStories + 1), '-dir', 6)
        ops.equalDOF(int(beam_link[i]), int(beam_hinge_nodes[i]), 1, 2)
        
    ops.region(2, '-eleRange', min(eID_beam_hinge), max(eID_beam_hinge))
    
    
    
    
    ###################################################################################################
    "Applying vertical load to frame to determine the building period"
    ###################################################################################################
    
    "Create the nodal load - command: load nodeID xForce yForce"
    
    # Gravity load
    opsplt.createODB("Nonlin_RCWall", "Gravity", Nmodes=3)
    
    # opsplt.plot_modeshape(1, 1, Model = "Nonlin_RCWall")
    
    
    ops.timeSeries('Linear', 1)  # applies the load in a linear manner (not all at once)
    ops.pattern("Plain", 1, 1) # create a plain load pattern - similar to ELF
    
    for i in range(0, len(main_nodes_vert)):
        "mass command is used to set the mass at each node"
        ops.load(int(main_nodes_vert[i]), 0, -mass_nodes[i]*9.81, 0)
    
    
    ops.recorder('Node', '-file', "results/node_disp/NLdisp_pin.out", '-closeOnWrite', '-node', 22, 23, 24, 25, '-dof', 1, 'disp')
    
    
    "(mode shape number, scale, ModelName - Displays the model saved in a database named"
    # # 
    # opsplt.plot_model("Nonlin_RCwall", show_nodes='yes', show_nodetags='yes', show_eletags='no', font_size=10, setview='3D', elementgroups=None, line_width=1, filename=None)
    
    # create DOF number
    ops.numberer("RCM")
    # create SOE
    ops.system('BandGeneral')
    # create constraint handler
    ops.constraints("Transformation")
    # create number of steps
    nsteps=25
    # create integrator
    ops.integrator('LoadControl', 1/nsteps)
    # create algorithm
    ops.test('RelativeEnergyIncr', 1e-1, 200, 0) # convergence scheme applied to the model
    ops.algorithm("KrylovNewton")                      # solution scheme - Newton-Raphson method of solving non-linear equations
    # create analysis object
    ops.analysis("Static")                       # analysis type - ie, static, transient etc..
    # perform the analysis
    # ops.recorder('Node', '-file', "results/modal/eigen.out",'-closeOnWrite','-dof',1,2,3,'eigen')
    
    ops.analyze(nsteps)
    
        
    opsplt.createODB("Nonlin_RCWall", "Gravity", Nmodes=3)
        
    # opsplt.plot_modeshape(1, 1, Model = "Nonlin_RCWall")
    
    
    # printModel()
    # opsv.plot_model(fig_wi_he=(20., 14.))
    
    ops.eigen('-genBandArpack', 1)
    ops.record
    ops.loadConst('-time', 0.0)
    ops.wipeAnalysis()
    
    a=ops.eigen(10)
    w1=np.sqrt(a[0])                       # angular frequency of first mode
    w2=np.sqrt(a[1])                       # angular frequency of second mode
    w3=np.sqrt(a[2])
    
    # print('T1 (s) = ' +  str(np.round(2*np.pi/w1, 5)))
    
    zeta=0.02                                  # RC frame model damping ratio
    a0    =zeta*2.0*w1*w3/(w1 + w3);  	        # mass damping coefficient based on first and second modes
    a1    =zeta*2.0/(w1 + w3);		        # stiffness damping coefficient based on first and second modes
    ops.rayleigh(a0, 0, 0, a1)
    
    
    def create_cntrlnodes(n):
        return [i for i in range(102, 102 + n - 1)]
    cntrlnodes = create_cntrlnodes(NStories + 1)
    
    # # print("cntrlnodes:", cntrlnodes)
    
    ms_1 = np.zeros([len(cntrlnodes)])
    for i in range(0, len(cntrlnodes)):
        ms_1[i] = ops.nodeEigenvector(cntrlnodes[i], 1, 1) # obtains the mode shape of the first mode
    ms_1 = ms_1/(ms_1[-1])                            # normalised the mode shape (sum equal to 1)
    ms_1 = np.round(ms_1, 3)   
    ms_1 = ms_1/sum(ms_1)                            # normalised the mode shape (sum equal to 1)
    
    # print(ms_1)
    
    
    # print('T2 (s) = ' +  str(np.round(2*np.pi/w2, 5)))
    
    # ms_2 = np.zeros([len(cntrlnodes)])
    # for i in range(0, len(cntrlnodes)):
    #     ms_2[i] = ops.nodeEigenvector(cntrlnodes[i], 2, 1)  # obtains the mode shape of the first mode
    # ms_2 = ms_2/(ms_2[-1])                                  # normalised the mode shape (sum equal to 1)
    # ms_2 = np.round(ms_2, 3)   
    
    # print(ms_2)

    # After the building model is set up, return the required variables
    return {
    'cntrlnodes': cntrlnodes,
    'ms_1': ms_1,
    'NBays': NBays,
    'NStories': NStories,
    'HBuilding': HBuilding,
    'mass_frame': mass_frame
    }

buildingmodel()