from ultralytics import YOLO
import cv2
import math
from rembg import remove 
from PIL import Image 
import time
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480) 

# model
model = YOLO("weights/best.pt")
# model = YOLO("yolo-Weights/yolov8x.mlpackage")
# print(model)
# model.export(format="coreml")



# object classes
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]

classNames = ["battery", "can", "cardboard_bowl", "cardboard_box", "chemical_plastic_gallon", "chemical_spray_can",
            "light_bulb" ,
          "paint_bucket" ,
           "plastic_bag" ,
        "plastic_bottle" ,
    "plastic_bottle_cap" ,
           "plastic_box" ,
           "plastic_cup" ,
       "plastic_cup_lid" ,
         "plastic_spoon" ,
           "scrap_paper" ,
         "scrap_plastic" ,
             "snack_bag" ,
                 "stick" ,
                 "straw" ,
        "toilet_cleaner"]

RECYCLABLE=['cardboard_box','can','plastic_bottle_cap','plastic_bottle','reuseable_paper','scrap_paper', 'plastic_bag']
NON_RECYCLABLE=['stick','plastic_cup','snack_bag','plastic_box','straw','plastic_cup_lid','scrap_plastic','cardboard_bowl','plastic_cultery']
HAZARDOUS=['battery','chemical_spray_can','chemical_plastic_bottle','chemical_plastic_gallon','light_bulb','paint_bucket']

while True:
    success, img = cap.read()
    imgSave = img.copy()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            status = ""
            if(RECYCLABLE.__contains__(classNames[cls])):
                print("Trash Status -> Recyclable")
                status = "Recyclable"
            elif(NON_RECYCLABLE.__contains__(classNames[cls])):
                print("Trash Status -> Non-Recyclable")
                status = "Non-Recyclable"
            elif(HAZARDOUS.__contains__(classNames[cls])):
                print("Trash Status -> Hazardous")
                status = "Hazardous"


            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls] + " " + status, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite("myImage.png", img)
        cv2.imwrite("myImage.jpg", img)
        break

cap.release()
cv2.destroyAllWindows()

# Store path of the image in the variable input_path 
input_path = 'myImage.png' 

# Store path of the output image in the variable output_path 
output_path = 'myImage.png' 

# Processing the image 
input = Image.open(input_path)

# Removing the background from the given Image
output = remove(input)

#Saving the image in the given path 
output.save(output_path)

# time.sleep(3)

from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="F5Zmam44xQVo6AqwQedf"
)

result = CLIENT.infer("myImage.png", model_id="material-recognition-2.0/5")

# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="F5Zmam44xQVo6AqwQedf"
# )

# result = CLIENT.infer("myImage.png", model_id="yolo-waste-detection/1")


# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="F5Zmam44xQVo6AqwQedf"
# )

# result = CLIENT.infer("myImage.jpg", model_id="yolo-detection-a034c/1")

# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="F5Zmam44xQVo6AqwQedf"
# )

# result = CLIENT.infer("myImage.png", model_id="g_project/1")

print(result.get("predictions"))
# print("Paper")

# if(result.get("predictions"))

centerX = int(result.get("predictions")[0].get("x"))
centerY = int(result.get("predictions")[0].get("y"))
width = int(int(result.get("predictions")[0].get("width")) / 2)
height = int(int(result.get("predictions")[0].get("height")) / 2)
p1 = (centerX - width, centerY - height)
p2 = (centerX + width, centerY + height)


curImage = cv2.imread("myImage.png")

curImage = cv2.rectangle(curImage, p1, p2, (255, 0, 0), 4)
cv2.imwrite("myImageResult.png", curImage)