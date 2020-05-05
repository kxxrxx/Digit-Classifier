from tkinter import *
from tkinter import ttk, filedialog, messagebox, Button, Label, Canvas

import cv2
import numpy as np
from PIL import ImageTk, Image, ImageGrab
from tensorflow import keras

class Model:
    def __init__(self):
        pass

    def predict(self):
        img = cv2.imread('image.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.astype('float32')
        img /= 255
        img = np.reshape(img, (1, 28, 28, 1))
        model = keras.models.load_model('model.h5')
        return model.predict_classes(img)[0]


class MainApp:
    def __init__(self):
        self.pre = None
        # Model object which will predict our number
        self.model = Model()
        # Create root window
        self.root = Tk()
        self.root.geometry("330x415")  # width x height: 330x415
        self.root.title("Digit Classifier")
        self.root.resizable(0, 0)

        # Creating GUI Elements
        tab_control = ttk.Notebook(self.root)
        tab1 = ttk.Frame(tab_control)
        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab1, text='Drawing')
        tab_control.add(tab2, text='Picture')
        tab_control.pack(expand=1, fill="both")

        self.lbl_draw = LabelFrame(tab1, text="Draw Here")
        self.lbl_draw.pack(fill=X)
        self.area_draw = Canvas(self.lbl_draw, width=308, height=308, bg='black')
        self.area_draw.bind('<B1-Motion>', self.draw)
        self.area_draw.pack(ipady=3)

        self.lbl_image = LabelFrame(tab2, text="Image")
        self.lbl_image.pack(side=RIGHT, fill=X)
        self.area_img = Label(self.lbl_image, image="", text="Select an image", padx=150, pady=150)
        self.area_img.pack()

        self.btn_clearD = Button(self.lbl_draw, text="Clear Drawing", bg="lightblue", command=self.clear)
        self.btn_clearD.pack()

        self.btn_load_img = Button(self.lbl_image, text="Browse", bg="lightblue", command=self.browse_image)
        self.btn_load_img.pack(in_=self.lbl_image, side=LEFT, fill=BOTH, expand=True)
        self.btn_clearP = Button(self.lbl_image, text="Clear Image", bg="lightblue", command=self.clear)
        self.btn_clearP.pack(in_=self.lbl_image, side=LEFT, fill=BOTH, expand=True)
        self.btn_camera_img = Button(self.lbl_image, text="Camera", bg="lightblue", command=self.open_camera)
        self.btn_camera_img.pack(in_=self.lbl_image, side=RIGHT, fill=BOTH, expand=True)

        self.area_predict = Frame(self.root)
        self.area_predict.pack(side=LEFT, fill=BOTH)

        self.btn_predict = Button(self.area_predict, text="Predict Digit", command=self.predict_digit, bg='brown',
                                  fg='white',
                                  font=('helvetica', 9, 'bold'))
        self.btn_predict.pack(in_=self.area_predict, side=LEFT)

        self.lbl_output = Label(self.area_predict)
        self.lbl_output.configure(text='''Predicted Digit:''')
        self.lbl_output.pack(side=LEFT, padx=25)

    def draw(self, event):
        self.area_draw.create_oval(event.x, event.y, event.x + 13, event.y + 13, outline='white', fill='white')
        self.area_draw.create_rectangle(event.x, event.y, event.x + 12, event.y + 12, outline='white', fill='white')
        self.pre = 'D'

    def run(self):
        self.root.mainloop()

    def clear(self):
        self.area_draw.delete('all')
        self.area_img.image = ""
        for widget in self.area_predict.winfo_children():
            widget.pack_forget()

        self.btn_predict.pack(in_=self.area_predict, side=LEFT)

        self.lbl_output = Label(self.area_predict)
        self.lbl_output.configure(text='''Predicted Digit:''')
        self.lbl_output.pack(side=LEFT, padx=25)

    def browse_image(self):
        try:
            self.area_img.text = ""
            file = filedialog.askopenfilename(filetypes=[('Images', ['*jpeg', '*png', '*jpg'])])
            file = Image.open(file)
            file = file.resize((315, 315))
            file.save('image.jpg')
            file = ImageTk.PhotoImage(file)
            self.area_img.configure(image=file)
            self.area_img.image = file
            self.pre = 'P'
        except Exception as e:
            messagebox.showerror("Error: Please try again.", e)

    def open_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Error: cannot open camera")

        while True:
            ret, frame = cap.read()
            cv2.rectangle(img=frame, pt1=(220, 140), pt2=(420, 340), color=(0, 0, 255), thickness=2)
            cv2.putText(frame, 'Fit digit within the rectangle then press ENTER', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.imshow('Input', frame)
            c = cv2.waitKey(1)
            ref_point = [[222, 142], [418, 338]]
            if c == 13:
                crop_img = frame[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
                cv2.imshow("crop_img", crop_img)
                cv2.imwrite('image.jpg', crop_img)
                file = Image.open('image.jpg')
                file = file.resize((320, 320))
                file = ImageTk.PhotoImage(file)
                self.area_img.configure(image=file)
                self.area_img.image = file
                break
        cap.release()
        cv2.destroyAllWindows()
        self.pre = 'P'

    def predict_digit(self):
        if self.pre is None:
            messagebox.showerror(title='No image or drawing found.',
                                 message="Draw, Browse or Capture image first.")
        else:
            if self.pre == 'D' or self.pre == 'P':
                x = self.root.winfo_rootx() + self.area_draw.winfo_x()
                y = self.root.winfo_rooty() + self.area_draw.winfo_y()
                x1 = x + self.area_draw.winfo_width()
                y1 = y + self.area_draw.winfo_height()
                ImageGrab.grab().crop((x, y + 20, x1, y1)).save('image.jpg')

                self.prediction = StringVar()
                self.lbl_prediction = Label(self.area_predict, textvariable=self.prediction)
                self.lbl_prediction.pack(side=LEFT, ipadx=0)
                self.prediction.set(self.model.predict())


if __name__ == "__main__":
    MainApp().run()
