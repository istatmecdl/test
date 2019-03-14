import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

import matplotlib
matplotlib.use('Agg')
import os

# File import
from Settings.settings import SetParameters
from Preprocessing.preprocessing import load_EuroSat_classify
from Initialization.initialization import Initialization
from Models.model_preparation import ModelPreparation
from Preprocessing.data_preparation import DataPreparation
from View.view import Viewclass
from enginecv_classify import Classify
from enginecv_train import Training

# Other imports
import platform

## INTERFACE imports
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
import pdb
from tkinter import ttk, font
import matplotlib.pyplot as plt 
from tkinter import filedialog as fd
from tkinter import messagebox as ms
import cv2
from time import sleep
import tkinter.ttk as ttk
import time



# CLASSIFY
class application:
    def __init__(self, master):
        self.master = master

        self.c_size = (700,500)
        self.setup_gui(self.c_size)
        self.img = None
   
    def setup_gui(self,s):
        self.canvas = Canvas(self.master, height = s[1], width = s[0], bg = 'white',bd = 2, relief = 'flat')
        self.ima = tk.PhotoImage(file="Images\\noima.gif")
        self.canvas.create_image(350, 255, image=self.ima, anchor=CENTER)
        self.canvas.pack(pady = 1)
        
        button_frame = tk.Frame(self.master)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.image01 = tk.PhotoImage(file="Images\\Immagine1.png")
        self.image02 = tk.PhotoImage(file="Images\\Immagine2.png")
        self.image03 = tk.PhotoImage(file="Images\\Immagine3.png")
        self.image04 = tk.PhotoImage(file="Images\\Immagine4.png")

        self.b01 = tk.Button(button_frame, image=self.image01, activebackground='gray70', bd = 0, bg = 'gray70', command = self.make_image)
        self.b02 = tk.Button(button_frame, image=self.image02, activebackground='gray70', bd = 0, bg = 'gray70', command = self.train_model)
        self.b03 = tk.Button(button_frame, image=self.image03, activebackground='gray70', bd = 0, bg = 'gray70', command = self.classify_image)
        self.b04 = tk.Button(button_frame, image=self.image04, activebackground='gray70', bd = 0, bg = 'gray70', command = self.pie)
        
        self.b01.config(bg="gray70")
        self.b02.config(bg="gray70")
        self.b03.config(bg="gray70")
        self.b04.config(bg="gray70")
        
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(0, weight=2)
        button_frame.columnconfigure(1, weight=2)

        self.b01.grid(row=0, column=0, ipady = 5, sticky=tk.W+tk.E)
        self.b02.grid(row=0, column=1, ipady = 5, sticky=tk.W+tk.E)        
        self.b03.grid(row=1, column=0, ipady = 5, sticky=tk.W+tk.E)
        self.b04.grid(row=1, column=1, ipady = 5, sticky=tk.W+tk.E)
        
        self.status=Label(self.master, text = 'Current Image: None', bg = 'ghost white', font = ('Arial Rounded MT',10), bd = 2, fg = 'black', relief = 'sunken', anchor = W)
        self.status.pack(side = BOTTOM, fill = X)
        
    def make_image(self):
        try:
            global File
            File = fd.askopenfilename()

            self.pilImage = Image.open(File)
            re=self.pilImage.resize((700,500),Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(re)
            self.canvas.delete(ALL)
            self.canvas.create_image(self.c_size[0]/2+10, self.c_size[1]/2+10, anchor=CENTER,image=self.img)
            self.status['text']='Current Image:'+ File

        except:
            ms.showerror('Error!','Import satellite image.')
    
    def train_model(self):
        try:
            def task():
                #history = Training.func_train()
                time.sleep(1)
                #global pie_path
               # pie_path = parameters.submission_path + '/CLASSIFY_Plot_pie.jpg'
                #ms.showinfo("END OF TRAIN/LOAD MODEL", "To continue view pie chart!")
                message.destroy()

                    #MESSAGE 02
                mess = Toplevel()
                mess.title("STEP 2: View training plot accuracy model")
                mess.wm_iconbitmap('Images\\logo01.ico')
                mess.resizable(0,0)

                canvas = Canvas(mess, width=800, height=600)
                canvas.pack()
                train_plot = tk.PhotoImage(file='..\\Output\\Metrics\\SAVE_CSV_TRAIN_Plot.png')
                train_plot = train_plot.subsample(7) 
                canvas.create_image(396,300,image=train_plot)
                canvas.image = train_plot

                msglab = tk.Label(mess, text = 'TRAINING\nThis model achieves 98.7% of Accuracy', bg = 'white')
                msglab.pack(ipadx=50, ipady=10, fill='both', expand=True)
                msglab.config(font=('Arial Rounded MT', 10))

                b = tk.Button(mess, text="OK", command=mess.destroy)
                b.pack()
                
            #MESSAGE 
            message = Toplevel()
            message.title("STEP 2: Training model")
            message.wm_iconbitmap('Images\\logo01.ico')
            message.resizable(0,0)
            
            canvas = Canvas(message, width=100, height=100)
            canvas.pack()
            img=tk.PhotoImage(file='Images\\train_model.png') 
            canvas.create_image(50,50,image=img)
            canvas.image = img

            msglab = tk.Label(message, text = 'Loading...', bg = 'white')
            msglab.pack(ipadx=50, ipady=10, fill='both', expand=True)
            msglab.config(font=('Arial Rounded MT', 10))

            #b = tk.Button(message, text="OK", command=message.destroy)
            #b.pack()
            
            message.after(2,task)
            message.mainloop()

        except:
            ms.showerror('Error!','There is no image.')

    def classify_image(self):
        def task():
            # TO recognize the File set interface = True in  engine.ini
            try:
                pie, bar, parameters = Classify.clas(File)
                global pie_path
                pie_path = parameters.submission_path + '/CLASSIFY_Plot_pie.jpg'
                #ms.showinfo("END OF TRAIN/LOAD MODEL", "To continue view pie chart!")
                message.destroy()
                
                #MESSAGE 02
                mex = Toplevel()
                mex.title("STEP 3: Classify")
                mex.wm_iconbitmap('Images\\logo01.ico')
                #mex.resizable(0,0)

                canvas = Canvas(mex, width=100, height=100)
                canvas.pack()
                train_plot = tk.PhotoImage(file='Images\\done.png')
                train_plot = train_plot.subsample(7) 
                canvas.create_image(50,50,image=train_plot)
                canvas.image = train_plot

                msglab = tk.Label(mex, text = 'Classification is done!', bg = 'white')
                msglab.pack(ipadx=50, ipady=10, fill='both', expand=True)
                msglab.config(font=('Arial Rounded MT', 10))

                b = tk.Button(mex, text="OK", command=mex.destroy)
                b.pack()
                
            except:
                ms.showerror('Error!','There is no image.')
        #MESSAGE
        message = Toplevel()
        message.title("STEP 3: Classify")
        message.wm_iconbitmap('Images\\logo01.ico')
        #message.resizable(0,0)
        
        canvas = Canvas(message, width=100, height=100)
        canvas.pack()
        img=tk.PhotoImage(file='Images\\classify.png') 
        canvas.create_image(50,50,image=img)
        canvas.image = img

        msglab = tk.Label(message, text = 'Loading...', bg = 'white')
        msglab.pack(ipadx=50, ipady=10, fill='both', expand=True)
        msglab.config(font=('Arial Rounded MT', 10))

        message.after(2,task)
        message.mainloop()
        
    def pie(self):
        try: 
            self.pilImage = Image.open(pie_path)
            re=self.pilImage.resize((700,500),Image.ANTIALIAS)
            self.img_pie = ImageTk.PhotoImage(re)
            self.canvas.delete(ALL)
            self.canvas.create_image(self.c_size[0]/2+10, self.c_size[1]/2+10, anchor=CENTER,image=self.img_pie)
            self.status['text']='Current Image:'+ pie_path
            
            #MESSAGE STEP ONE
            message = Toplevel()
            message.title("STEP 4: View Statistics")
            message.wm_iconbitmap('Images\\logo01.ico')
            message.resizable(0,0)
            
            canvas = Canvas(message, width=100, height=100)
            canvas.pack()
            img=tk.PhotoImage(file='Images\\statistics.png') 
            canvas.create_image(50,50,image=img)
            canvas.image = img

            msglab = tk.Label(message, text = 'END OF PROCESS!', bg = 'white')
            msglab.pack(ipadx=50, ipady=10, fill='both', expand=True)
            msglab.config(font=('Arial Rounded MT', 10))
            
            b = tk.Button(message, text="OK", command=message.destroy)
            b.pack()
            
            #ms.showinfo("END OF ANALYZE PROCESS", "PIE CHART RESULTS!!")
        except:
            ms.showerror('Error!','Import image and then classify\n to view statistics!')

def created_by():
    top = Toplevel()
    top.title("About this application...")
    top.wm_iconbitmap('Images\\logo01.ico')
    top.resizable(0,0)
    
    canvas = Canvas(top, width=100, height=100)
    canvas.pack()
    img=tk.PhotoImage(file='Images\\logores.png') 
    canvas.create_image(50,50,image=img)
    canvas.image = img

    msglab = tk.Label(top, text = 'Created by:\n\nFrancesco Pugliese - ISTAT researcher\nand\nEleonora Bernasconi - ISTAT intern', bg = 'white')
    msglab.pack(ipadx=50, ipady=10, fill='both', expand=True)
    msglab.config(font=('Arial Rounded MT', 10))
    
    b = tk.Button(top, text="OK", command=top.destroy)
    b.pack()

root=Tk()

#BACKGROUND
background_label = Label(root, bg = 'white')
background_label.place(x=0, y=0, relwidth=1, relheight=1)

#MENU
menu = Menu(root, bg='white',  tearoff=False)

helpmenu = Menu(menu, tearoff=False)
menu.add_cascade(label='About', menu=helpmenu)
helpmenu.add_command(label='Created by', command = created_by)

root.wm_iconbitmap('Images\\logo01.ico')
root.title('Automatic Extraction of Statistics in SATELLITE IMAGERY by DEEP LEARNING')
application(root)
root.resizable(0,0)
root.config(menu=menu)
root.mainloop()

