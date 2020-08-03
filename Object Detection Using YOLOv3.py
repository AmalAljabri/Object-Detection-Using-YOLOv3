from tkinter import *
from tkinter.filedialog import askopenfile
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, Image, ImageDraw
weights_path=r"D:\Object Detection Using YOLOv3\yolov3.weights"
config_path=r"D:\Object Detection Using YOLOv3\yolov3.cfg"
net = cv2.dnn.readNet(config_path,weights_path)

classes = [['person','شخص'], ['bicycle','دراجة'], ['car','سيارة'], ['motorcycle','دراجة نارية'],
           [ 'airplane','طائرة'], ['bus','حافلة'], ['train','قطار'], ['truck','شاحنة'],
           ['boat','قارب'],[ 'traffic light','إشارات المرور'], ['fire hydrant','صنبور النار'], ['stop sign','علامة التوقف'],
           ['parking meter','عداد وقوف السيارات'],['bench','مقعد'], ['bird','طائر'],
           ['cat','قطة'], ['dog','كلب'], ['horse','حصان'],[ 'sheep','خروف'], ['cow','بقرة'], ['elephant','فيل'], ['bear','دب'],
           ['zebra','حمار وحشي'], ['giraffe','زرافة'], ['backpack','حقيبة ظهر'],
           ['umbrella','مظلة'], ['handbag','حقيبة يد'], ['tie','ربطة عنق'], ['suitcase','حقيبة سفر'],
           ['frisbee','الفريسبي'], ['skis','الزلاجات'], ['snowboard','لوح التزلج'],[ 'sports ball','كرة'],
           ['kite','طائرة ورقية'],[ 'baseball bat','مضرب بيسبول'], ['baseball glove','قفاز البيسبول'], ['skateboard','لوح التزلج'],
           ['surfboard','لوح ركوب الأمواج'], ['tennis racket','مضرب التنس'],['bottle','زجاجة'],[ 'wine glass','كأس'],
           ['cup','كأس'], ['fork','شوكة'],[ 'knife','سكين'], ['spoon','ملعقة'], ['bowl','وعاء'],[ 'banana','موز'], ['apple','تفاح'],
           ['sandwich','ساندويتش'],['orange','برتقال'], ['broccoli','بروكلي'],[ 'carrot','جزر'], ['hot dog','هوت دوج'],
           ['pizza','بيتزا'], ['donut','دونات'],[ 'cake','كيك'], ['chair','كرسي'], ['couch','أريكة'],[ 'potted plant','نبات'],
           ['bed','سرير'], ['dining table','طاولة طعام'], ['toilet','الحمام'], ['tv','تلفزيون'], ['laptop','كمبيوتر محمول'],
           ['mouse','ماوس'],[ 'remote','ريموت'], ['keyboard','لوحة مفاتيح'], ['cell phone','جوال'],[ 'microwave','ميكروويف'],
           ['oven','فرن'], ['toaster','محمصة'],[ 'sink','مغسلة'], ['refrigerator','ثلاجة'], ['book','كتاب'], ['clock','ساعة'],
           ['vase','مزهرية'], ['scissors','مقص'], ['teddy bear','دمية'], ['hair drier','مجفف شعر'], ['toothbrush','فرشاة أسنان']]
#print(classes)

root = Tk()  
root.geometry("600x500+250+30")  
root.resizable(0, 0)
root.title("Object Detection Using YOLOv3")
root.configure(background="#f0f0f0")



def Upload_Image(): 
    file = asfile = askopenfile(title='Upload Image', filetypes=[('Image Files', ['.jpeg', '.jpg', '.png', '.gif', '.tiff', '.tif', '.bmp'])])
    if file is not None:  
        #print(file.name)
        image = cv2.imread(file.name)
        text = ''    
        image=cv2.resize(image,(600,500))
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        output_layers = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in output_layers:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            text = str(classes[class_ids[i]][1])
            font = ImageFont.truetype("font.ttf", 35)
            reshaped_text = arabic_reshaper.reshape(text)   
            arabic_text = get_display(reshaped_text)
            #print(arabic_text)
            image_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(image_pil)
            draw.text((round(x),round(y)-50), arabic_text, (255,255,255),  font=font)
            image = np.array(image_pil)
            cv2.rectangle(image, (round(x),round(y)), (round(x+w), round(y+h)), (255,255,255), 2)
            #cv2.putText(image, "{:.0f}%".format(confidences[i]*100), (round(x),round(y)-40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2)
            
        cv2.imshow("Object Detection Using YOLOv3", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
  

def Camera():
        camera = cv2.VideoCapture(0)
        while True:
            value,frame = camera.read()
            frame =cv2.resize(frame,(600,500))
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
            net.setInput(blob)
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            output_layers = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            key = cv2.waitKey(1)
            for out in output_layers:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] *frame.shape[1])
                        h = int(detection[3] * frame.shape[0])
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
                    
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
     
            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                text = str(classes[class_ids[i]][1])
                font = ImageFont.truetype("font.ttf", 35)
                reshaped_text = arabic_reshaper.reshape(text)   
                arabic_text = get_display(reshaped_text)
                #print(arabic_text)
                image_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(image_pil)
                draw.text((round(x),round(y)-50), arabic_text, (255,255,255),  font=font)
                frame = np.array(image_pil)
                cv2.rectangle(frame, (round(x),round(y)), (round(x+w), round(y+h)), (255,255,255), 2)
                #cv2.putText(frame, "{:.0f}%".format(confidences[i]*100), (round(x),round(y)-40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2)          
            cv2.imshow("Object Detection Using YOLOv3", frame)
            if key == 27:
                break

        cv2.destroyAllWindows()
        camera.release()
        
global img
Label1 = Label(root)
Label1.place(relx=0.0, rely=0.37, height=160, width=600)
img = PhotoImage(file = r"D:\Object Detection Using YOLOv3\yolo.png")
Label1.configure(image=img)

Label1 = Label(root)
Label1.place(relx=0.0, rely=0.07, height=120, width=600)
Label1.configure(background="#f0f0f0")
Label1.configure(borderwidth="5")
Label1.configure(font="-family {Montserrat Black*} -size 40 -weight bold ")
Label1.configure(foreground="#000000")
Label1.configure(text='''اكتشاف الأشياء\n باستخدام خوارزمية ''')

Button1 = Button(root)
Button1.place(relx=0.55, rely=0.8, height=55, width=180)
Button1.configure(background="#18e1e4")
Button1.configure(command=Upload_Image)
Button1.configure(font="-family {Montserrat Black*} -size 30 -weight bold ")
Button1.configure(foreground="#000000")
Button1.configure(relief="ridge")
Button1.configure(anchor='center')
Button1.configure(text='''صورة''')

Button2 = Button(root)
Button2.place(relx=0.15, rely=0.8, height=55, width=180)
Button2.configure(background="#18e1e4")
Button2.configure(command=Camera)
Button2.configure(font="-family {Montserrat Black*} -size 30 -weight bold ")
Button2.configure(foreground="#000000")
Button2.configure(relief="ridge")
Button2.configure(anchor='center')
Button2.configure(text='''كاميرا''')

root.mainloop()

