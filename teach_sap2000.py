import configparser
import os
import sys
import comtypes.client
import numpy as np

def get_point_displacement(nodes,SapModel):
    displacements = []
    ObjectElm = 0
    NumberResults = 0
    m001 = []
    result = []
    Obj = []
    Elm = []
    ACase = []
    StepType = []
    StepNum = []
    U1 = []
    U2 = []
    U3 = []
    R1 = []
    R2 = []
    R3 = []
    ObjectElm = 0
    [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3,ret] = SapModel.Results.JointDispl(nodes, ObjectElm, NumberResults, Obj,Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
    return Obj,U1, U2, U3, R1, R2, R3

def get_frame_reactions(frames,SapModel):
    result = []
    Object11 = 0
    Obj = []
    ObjSta = []
    Elm = []
    ElmSta = []
    LoadCase = []
    StepType = []
    StepNum = []
    NumberResults = 0
    P = []
    V2 = []
    V3 = []
    T = []
    M2 = []
    M3 = []
    [NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3,
     ret] = SapModel.Results.FrameForce(frames, Object11, NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2,M3)
    return Obj, ObjSta,P, V2, V3, T, M2,M3



APIPath = os.path.join(os.getcwd(), 'cases')
cfg = configparser.ConfigParser()
cfg.read("Configuration.ini", encoding='utf-8')
ProgramPath = cfg['SAP2000PATH']['dirpath']
if not os.path.exists(APIPath):
    try:
        os.makedirs(APIPath)
    except OSError:
        pass

AttachToInstance = True
SpecifyPath = False

# ModelPath = os.path.join(APIPath, 'API_1-001.sdb')
helper = comtypes.client.CreateObject('SAP2000v1.Helper')
helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
if AttachToInstance:
    # attach to a running instance of SAP2000
    try:
        # get the active SapObject
        mySapObject = helper.Getobject("CSI.SAP2000.API.SapObject")
    except (OSError, comtypes.COMError):
        print("No running instance of the program found or failed to attach.")
        sys.exit(-1)
else:
    if SpecifyPath:
        try:
            # 'create an instance of the SAPObject from the specified path
            mySapObject = helper.CreateObject(ProgramPath)
        except (OSError, comtypes.COMError):
            print("Cannot start a new instance of the program from" + ProgramPath)
            sys.exit(-1)
    else:
        try:
            # create an instance of the SapObject from the latest installed SAP2000
            mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
        except (OSError, comtypes.COMError):
            print("Cannot start a new instance of the program")
            sys.exit(-1)

    # start SAP2000 application
    mySapObject.ApplicationStart()

# create SapModel object
SapModel = mySapObject.SapModel
# initialize model
SapModel.InitializeNewModel()
ModelPath = os.path.join(APIPath, 'API_1-001.sdb')

ret = SapModel.File.NewBlank()

# switch units
N_mm_C = 9
ret = SapModel.SetPresentUnits(N_mm_C)

# define material property

MATERIAL_CONCRETE = 2

ret = SapModel.PropMaterial.SetMaterial('CONC', MATERIAL_CONCRETE)

# assign isotropic mechanical properties to material

ret = SapModel.PropMaterial.SetMPIsotropic('CONC', 3600, 0.2, 0.0000055)

# define rectangular frame section property

ret = SapModel.PropFrame.SetRectangle('R1', 'CONC', 12, 12)


point = [[0,0,0],[3000,0,0],[3000,8000,0],[0,8000,0],[0,0,4000],[3000,0,4000],[3000,8000,4000],[0,8000,4000]]
frame = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
plane = [[4,5,6,7]]
for node_indx in range(len(point)):
    x, y, z = point[node_indx]
    ret = SapModel.PointObj.AddCartesian(x, y, z, None, "nodes" + str(node_indx), "Global")

""" define frames """

for edge_indx in range(len(frame)):
    indx1, indx2 = frame[edge_indx]
    Point1 = "nodes" + str(indx1)
    Point2 = "nodes" + str(indx2)
    name = "frame_" + str(edge_indx)
    section_name = 'R1'
    ret = SapModel.FrameObj.AddByPoint(Point1, Point2, " ", section_name, name)
    if edge_indx in range(4,8):
        ret = SapModel.FrameObj.SetLoadDistributed(name, "DEAD", 1, 10, 0, 1, 4.2, 4.2)
ret = SapModel.PropArea.SetShell_1("Plane0", 1, True, "4000Psi", 0, 0, 0)
""" define frames """
for plane_indx in range(len(plane)):
    plane_node = []
    for node_indx in range(len(plane[plane_indx])):
        plane_node.append(point[plane[plane_indx][node_indx]])
    plane_node = np.array(plane_node)
    node_x = np.array(plane_node)[:, 0].tolist()
    node_y = np.array(plane_node)[:, 1].tolist()
    node_z = np.array(plane_node)[:, 2].tolist()
    ret = SapModel.AreaObj.AddByCoord(len(plane[plane_indx]), node_x, node_y,
                                      node_z, 'Plane0', "Default",
                                      f"plane_{node_indx}_{plane_indx}bottom", "Global")

    ret = SapModel.AreaObj.SetLoadUniformToFrame( f"plane_{node_indx}_{plane_indx}bottom", "DEAD", -0.0012, 9,
                                                 2, True, "Global")

ret = SapModel.LoadPatterns.Add('LIVE', 3, 0, True)

ret = SapModel.RespCombo.Add("COMB1", 0)
ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "DEAD", 1.3)
ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "LIVE", 1.5)

res1 = [True, True, True, True, True, True]
for edge_indx in range(4):
    Point1 = "nodes" + str(edge_indx)
    ret = SapModel.PointObj.setRestraint(Point1, res1)

ret = SapModel.File.Save(ModelPath)
ret = SapModel.Analyze.RunAnalysis()

""" results output """
ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
# ret = SapModel.Results.Setup.SetCaseSelectedForOutput("DEAD")

ret = SapModel.Results.Setup.SetComboSelectedForOutput("COMB1")

Obj,U1, U2, U3, R1, R2, R3 =get_point_displacement('nodes4',SapModel)
Obj, ObjSta,P, V2, V3, T, M2,M3 =get_frame_reactions('frame_4',SapModel)