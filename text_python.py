import os

import sys

import comtypes.client

# set the following flag to True to attach to an existing instance of the program

# otherwise a new instance of the program will be started

AttachToInstance = True

# set the following flag to True to manually specify the path to SAP2000.exe

# this allows for a connection to a version of SAP2000 other than the latest installation

# otherwise the latest installed version of SAP2000 will be launched

SpecifyPath = False

# if the above flag is set to True, specify the path to SAP2000 below

ProgramPath = 'D:\sap200023\SAP2000.exe'


# full path to the model

# set it to the desired path of your model

APIPath = 'D:\desktop\姜佳琦'

if not os.path.exists(APIPath):

    try:

        os.makedirs(APIPath)

    except OSError:

        pass

ModelPath = APIPath + os.sep + 'API_1-001.sdb'

# create API helper object

helper = comtypes.client.CreateObject('SAP2000v1.Helper')

helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)

if AttachToInstance:

    # attach to a running instance of SAP2000

    try:

        # get the active SapObject

        mySapObject = helper.GetObject("CSI.SAP2000.API.SapObject")

    except (OSError, comtypes.COMError):

        print("No running instance of the program found or failed to attach.")

        sys.exit(-1)

else:

    if SpecifyPath:

        try:

            # 'create an instance of the SAPObject from the specified path

            mySapObject = helper.CreateObject(ProgramPath)

        except (OSError, comtypes.COMError):

            print("Cannot start a new instance of the program from " + ProgramPath)

            sys.exit(-1)

    else:

        try:

            # create an instance of the SAPObject from the latest installed SAP2000

            mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")

        except (OSError, comtypes.COMError):

            print("Cannot start a new instance of the program.")

            sys.exit(-1)

    # start SAP2000 application

    mySapObject.ApplicationStart()

# create SapModel object

SapModel = mySapObject.SapModel

# initialize model

SapModel.InitializeNewModel()

# create new blank model

ret = SapModel.File.NewBlank()

# define material property

MATERIAL_CONCRETE = 2

ret = SapModel.PropMaterial.SetMaterial('CONC', MATERIAL_CONCRETE)

# assign isotropic mechanical properties to material

ret = SapModel.PropMaterial.SetMPIsotropic('CONC', 3600, 0.2, 0.0000055)

# define rectangular frame section property

ret = SapModel.PropFrame.SetRectangle('R1', 'CONC', 12, 12)

# define frame section property modifiers

ModValue = [1000, 0, 0, 1, 1, 1, 1, 1]

ret = SapModel.PropFrame.SetModifiers('R1', ModValue)

# switch to k-ft units

kip_ft_F = 4

ret = SapModel.SetPresentUnits(kip_ft_F)
