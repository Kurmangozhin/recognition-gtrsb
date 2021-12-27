import cv2
import numpy as np
import matplotlib.pyplot as plt



class_names = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing for vehicles over 3.5 metric tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Vehicles over 3.5 metric tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End of no passing by vehicles over 3.5 metric tons' }

           
onnx_model_name_pytorch = "model.onnx"

net = cv2.dnn.readNetFromONNX(onnx_model_name_pytorch)



mean = np.array([0.485, 0.456, 0.406]) * 255.0
scale = 1 / 255.0
std = [0.229, 0.224, 0.225]



input_img = cv2.imread('image/1.jpg', cv2.IMREAD_COLOR)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
plt.imshow(input_img)
input_img = input_img.astype(np.float32)



input_blob = cv2.dnn.blobFromImage(
    image = input_img,
    scalefactor = scale,
    size = (30, 30),  # img target size
    mean = mean,
    swapRB = True,    # BGR -> RGB
    crop = False      # center crop
)
input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)


# predictions
net.setInput(input_blob)
out = net.forward()

out_predictions = np.argmax(out[0], axis=0)
classes = class_names[out_predictions]
print(classes, out_predictions)
