from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import cv2
import json
from matplotlib import pyplot

juan_test = cv2.imread("juan_test.jpg")
coqui_test = cv2.imread("test_photos/coqui_test.jpg")
pajaro_test = cv2.imread("test_photos/pajaro_test.jpg")

apikey = 'iML6syjz13VDczCJhBYK-b_d6BlOWld01qn-fnUVAhtG'
authenticator = IAMAuthenticator(apikey)

visual_recognition = VisualRecognitionV3("2018-03-19", authenticator=authenticator)

with open('juan_test.jpg', 'rb') as images_file:
    classes = visual_recognition.classify(
        images_file,
        threshold='0.5',
        classifier_ids='DefaultCustomModel_1309469331').get_result()
print(json.dumps(classes, indent=2))
