from tkinter import *
import os
window=Tk()

C = Canvas(window, bg="blue", height=250, width=300)
filename = PhotoImage(file = "C:\\studentproj\\counter-with-yolo-and-sort-master\\o.png")
background_label = Label(window, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

def query():
    os.system('python main2.py')

btn=Button(window, text="START", fg='black', command=query)
btn.config( height = 3, width = 20 )
btn.place(x=320, y=400)

lbl=Label(window, text="QUERY ORIENTED MULTI-VIEW\n BOUNDING BOX REGRESSION MODEL\n BASED ON REAL-TIME DYNAMIC VEHICLE\n DETECTION AND TRACKING \n USING MACHINE LEARNING TECHNIQUES", fg='brown4', font=("Helvetica", 24))
lbl.place(x=50, y=50)

#########################################################



#########################################################
window.title('Traffic Counter')
window.geometry("800x600+10+10")
window.mainloop()