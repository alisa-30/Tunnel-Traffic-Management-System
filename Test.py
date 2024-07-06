import pixellib
import time
from pixellib.torchbackend.instance import instanceSegmentation

# firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
cred = credentials.Certificate('nodemcu-350de-firebase-adminsdk-j2j7a-41689d8693.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {'databaseURL': 'https://nodemcu-350de-default-rtdb.firebaseio.com'})

# As an admin, the app has access to read and write all data, regradless of Security Rules
led = db.reference()
ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl",detection_speed="fast",confidence=0.6)
target_classes = ins.select_target_classes(car=True,truck=True)
results,output=ins.segmentBatch("test", show_bboxes=True,output_folder_name="outputfolder",segment_target_classes = target_classes)
tool=['car','truck','bus']
# result to firebase
for i in range(len(results)):
    x=results[i].get("class_names")
    if any(value in x for value in tool):
         led.set({'LED':1})
         print("1")
         time.sleep(1)
    else:
        led.set({'LED':0})
        print("0")
        time.sleep(1)
