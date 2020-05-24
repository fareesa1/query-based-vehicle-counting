import tkinter as tk

from tkinter import *

try:
    import os
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog

root = tk.Tk()
# C = Canvas(root, bg="blue", height=250, width=300)
# filename = PhotoImage(file="C:\\studentproj\\counter-with-yolo-and-sort-master\\o.png")
# background_label = Label(root, image=filename)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)
#
# theLabel = Label(root, text="QUERY ORIENTED MULTI-VIEW DYNAMIC VEHICLE DETECTION AND TRACKING", bg="brown4", fg="white")
# theLabel.pack(fill=X)
#
# style = ttk.Style(root)
# style.theme_use("clam")



def c_open_file_old():
    rep = filedialog.askopenfilenames(
        parent=root,
        initialdir='/',
        initialfile='tmp',
        filetypes=[
            ("MP4", "*.mp4"),
            ("AVI", "*.avi"),
            ("All files", "*")])
    return rep
    # try:
    #     os.startfile(x[0])
    # except IndexError:
    #     print("No file selected")

c = c_open_file_old()
print(c)


def convertTuple(tup):
    str = ''.join(tup)
    return str


# Driver code
tuple = c
str = convertTuple(tuple)


input_video = str
output_video = "C:\studentproj\score.avi"
coco_weights  = "C:\studentproj\counter-with-yolo-and-sort-master\yolo-coco\yolov3.weights"
coco_names = "C:\studentproj\counter-with-yolo-and-sort-master\yolo-coco\coco.names"
coco_cfg = "C:\studentproj\counter-with-yolo-and-sort-master\yolo-coco\yolov3.cfg"
def query():
    os.system('python gui2.py')

def run_main():
    os.system('python main.py -o C:\studentproj\counter-with-yolo-and-sort-master\count.avi -y C:\studentproj\counter-with-yolo-and-sort-master\yolo-coco')

# frame = tk.Frame(root)
# frame.pack()
#
# bottomFrame = Frame(root)
# bottomFrame.pack(side=BOTTOM)
# button1 = tk.Button(frame, text="SELECT FILE", fg="BLACK", command=c_open_file_old)
#
# button1.config(height=3, width=20)
# button1.place(x=220, y=220)
# button1.pack()
#
lbl = Label(root,
            text="You have sucessfully updated the input file\n close this prompt to update query\n and process the input file",
            fg='brown4', font=("Helvetica", 12))
lbl.pack()
#
# theLabel = Label(root, text="ENTER QUERY: VEHICLE TYPE - CAR, BUS, TRUCK, MOTORBIKE, BICYCLE, HEAVY-VEHICLES ",
#                  bg="brown4", fg="white")
# theLabel.pack(fill=X)
#
#
# e1 = Entry(root).place(x=350, y=420)
# button2 = tk.Button(bottomFrame, text="QUIT", fg="red", command=quit)
#
# button2.pack(side=tk.BOTTOM)
#
# button3 = Button(bottomFrame, text="NEXT", command=run_main)
#
# button2.config(height=3, width=20)
# button3.config(height=3, width=20)
# button2.pack(side=RIGHT)
# button3.pack(side=LEFT)

# print(c_open_file_old().rep)
root.geometry("300x100+10+10")
root.mainloop()

