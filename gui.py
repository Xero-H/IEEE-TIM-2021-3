import tkinter as tk

top = tk.Tk()

top.title("UCI_HAR")
width = 260
height = 380
top.geometry(f'{width}x{height}')

x = "22.5"


var_Walking = tk.StringVar()
var_Upstair = tk.StringVar()
var_Downstairs = tk.StringVar()
var_Sitting = tk.StringVar()
var_Standing = tk.StringVar()
var_Laying = tk.StringVar()

var_Walking.set("?")
var_Upstair.set("?")
var_Downstairs.set("?")
var_Sitting.set("?")
var_Standing.set("?")
var_Laying.set("?")

text_Walking = tk.Label(top, text="Walking").place(x=50, y=0)
text_Upstairs = tk.Label(top, text="Walking Upstairs").place(x=50, y=20)
text_Downstairs = tk.Label(top, text="Walking Downstairs").place(x=50, y=40)
text_Sitting = tk.Label(top, text="Sitting").place(x=50, y=60)
text_Standing = tk.Label(top, text="Standing").place(x=50, y=80)
text_Laying = tk.Label(top, text="Laying").place(x=50, y=100)

text_Walking_value = tk.Label(top, textvariable=var_Walking).place(x=200, y=0)
text_Upstairs_value = tk.Label(top, textvariable=var_Upstair).place(x=200, y=20)
text_Downstairs_value = tk.Label(top, textvariable=var_Downstairs).place(x=200, y=40)
text_Sitting_value = tk.Label(top, textvariable=var_Sitting).place(x=200, y=60)
text_Standing_value = tk.Label(top, textvariable=var_Standing).place(x=200, y=80)
text_Laying_value = tk.Label(top, textvariable=var_Laying).place(x=200, y=100)


img_gif = tk.PhotoImage(file='./动作/问号.gif')
img_gif0 = tk.PhotoImage(file='./动作/走.gif')
img_gif1 = tk.PhotoImage(file='./动作/上楼.gif')
img_gif2 = tk.PhotoImage(file='./动作/下楼.gif')
img_gif3 = tk.PhotoImage(file='./动作/坐.gif')
img_gif4 = tk.PhotoImage(file='./动作/站立.gif')
img_gif5 = tk.PhotoImage(file='./动作/躺.gif')



label_img = tk.Label(top, image=img_gif)
label_img.place(x=30, y=120)


def Predict_result(prob, result):
    var_Walking.set(prob[0])
    var_Upstair.set(prob[1])
    var_Downstairs.set(prob[2])
    var_Sitting.set(prob[3])
    var_Standing.set(prob[4])
    var_Laying.set(prob[5])
    if result == 0:
        label_img.configure(image=img_gif0)
    elif result == 0:
        label_img.configure(image=img_gif1)
    elif result == 0:
        label_img.configure(image=img_gif2)
    elif result == 0:
        label_img.configure(image=img_gif3)
    elif result == 0:
        label_img.configure(image=img_gif4)
    elif result == 0:
        label_img.configure(image=img_gif5)

def Clear_result():
    var_Walking.set("?")
    var_Upstair.set("?")
    var_Downstairs.set("?")
    var_Sitting.set("?")
    var_Standing.set("?")
    var_Laying.set("?")
    label_img.configure(image=img_gif)

prob = [0,0,1,0,0,0]
result = 2
button = tk.Button(top, text='Prediction', command=Predict_result(prob, result))
button.place(x=60, y=330)

button_Clear = tk.Button(top, text='Clear', command=Clear_result)
button_Clear.place(x=150, y=330)

top.mainloop()
