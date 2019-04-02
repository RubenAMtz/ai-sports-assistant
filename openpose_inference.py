import sys
import cv2

# Import Openpose (Windows/Ubuntu/OSX)
PYOPENPOSE_DIR = "C:/Users/ruben/Documents/Github/openpose/build/python/openpose/Release"
MODELS_DIR = "C:/Users/ruben/Documents/Github/openpose/models/"

KEYPOINTS = ['nose', 'neck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder','lelbow', 
            'lwrist', 'midhip', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle',
            'reye', 'leye', 'rear', 'lear', 'lbigtoe', 'lsmalltoe', 'lheel', 'rbigtoe',
            'rsmalltoe', 'rheel']

sys.path.append(PYOPENPOSE_DIR)
import pyopenpose as op

#############################
## create csv out of openpose
#############################
params = {
    'model_folder': MODELS_DIR,
    'number_people_max': 1
}



# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()



def run_openpose(frame):

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    return datum.poseKeypoints.copy()
