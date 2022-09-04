import sys
sys.path.insert(1, "D:/Workspace/DATN/code")

from tkinter import *
import PIL.Image, PIL.ImageTk
import tkinter.filedialog
from tkinter import messagebox
from utils.utils import get_collection
from get_hand_info import get_hand_info_from_image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter
canvas = None

def get_right_hand_info1():
    global canvas, right_hand_info1
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.tiff .jpg .png')])
    if path is None:
        return
    right_hand_info1, fig = get_hand_info_from_image(path, username.get())
    canvas = FigureCanvasTkAgg(fig, left_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()
    messagebox.showinfo("Thành công", "Lấy thông tin bàn tay phải người dùng thành công")

def get_right_hand_info2():
    global canvas, right_hand_info2
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.tiff .jpg .png')])
    if path is None:
        return
    right_hand_info2, fig = get_hand_info_from_image(path, username.get())
    canvas = FigureCanvasTkAgg(fig, left_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()
    messagebox.showinfo("Thành công", "Lấy thông tin bàn tay phải người dùng thành công")

def get_right_hand_info3():
    global canvas, right_hand_info3
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.tiff .jpg .png')])
    if path is None:
        return
    right_hand_info3, fig = get_hand_info_from_image(path, username.get())
    canvas = FigureCanvasTkAgg(fig, left_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()
    messagebox.showinfo("Thành công", "Lấy thông tin bàn tay phải người dùng thành công")

def get_left_hand_info1():
    global canvas, left_hand_info1
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.tiff .jpg .png')])
    if path is None:
        return
    left_hand_info1, fig = get_hand_info_from_image(path, username.get())
    canvas = FigureCanvasTkAgg(fig, left_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()
    messagebox.showinfo("Thành công", "Lấy thông tin bàn tay trái người dùng thành công")

def get_left_hand_info2():
    global canvas, left_hand_info2
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.tiff .jpg .png')])
    if path is None:
        return
    left_hand_info2, fig = get_hand_info_from_image(path, username.get())
    canvas = FigureCanvasTkAgg(fig, left_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()
    messagebox.showinfo("Thành công", "Lấy thông tin bàn tay trái người dùng thành công")

def get_left_hand_info3():
    global canvas, left_hand_info3
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.tiff .jpg .png')])
    if path is None:
        return
    left_hand_info3, fig = get_hand_info_from_image(path, username.get())
    canvas = FigureCanvasTkAgg(fig, left_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()
    messagebox.showinfo("Thành công", "Lấy thông tin bàn tay trái người dùng thành công")

def userRegister():
    my_col = get_collection("users")

    name = username.get()
    address = user_address.get()
    email = user_email.get()
    gender = user_gender.get()
    phoneNumber = user_phoneNumber.get()
    if name == "":
        messagebox.showwarning('Cảnh báo',"Chưa điền tên người dùng")
        return

    if address == "":
        messagebox.showwarning('Cảnh báo',"Chưa điền địa chỉ người dùng")
        return

    if gender != "Nam" and gender != "Nữ":
        messagebox.showwarning('Cảnh báo',"Giới tính phải là Nam hoặc Nữ")
        return

    if email == "":
        messagebox.showwarning('Cảnh báo',"Chưa điền email người dùng")
        return
    if phoneNumber == "":
        messagebox.showwarning('Cảnh báo',"Chưa điền số điện thoại người dùng")
        return
    user_dict = {"name": name, "address": address, "gender": gender, "email": email, "phoneNumber": phoneNumber, "right_hand1": right_hand_info1, "right_hand2": right_hand_info2, "right_hand3": right_hand_info3, "left_hand1": left_hand_info1, "left_hand2": left_hand_info2, "left_hand3": left_hand_info3}
    try:
        my_col.insert_one(user_dict)
        messagebox.showinfo('Thành công',"User added successfully")
    except:
        messagebox.showinfo("Lỗi","Can't add data into Database")

def addUser(): 

    global username, user_address, user_gender, user_phoneNumber, user_email, root, left_frame
    root = Toplevel()
    root.title("Hệ thống chứng thực người")
    root.geometry("960x540")
    # root.resizable(0, 0)
    # root.state("zoomed")
    # Canvas1 = Canvas(root)
    
    # Canvas1.config(bg="#c6e2ff")
    # Canvas1.pack(expand=True,fill=BOTH)
    left_frame = Frame(root, bg="white",bd=5)
    left_frame.place(relx=0.3,rely=0.62,relwidth=0.4,relheight=0.26)
          
    headingFrame1 = Frame(root,bg="#FFBB00",bd=5)
    headingFrame1.place(relx=0.25,rely=0.05,relwidth=0.5,relheight=0.13)

    headingLabel = Label(headingFrame1, text="Thêm người dùng", bg='#f5f0f0', fg='black', font=('Courier',15))
    headingLabel.place(relx=0,rely=0, relwidth=1, relheight=1)


    labelFrame = Frame(root,bg='#f5f0f0')
    labelFrame.place(relx=0.1,rely=0.2,relwidth=0.8,relheight=0.4)
       
    # User Name
    lb1 = Label(labelFrame,text="Tên : ", bg='#f5f0f0', fg='black')
    lb1.place(relx=0.05,rely=0.03, relheight=0.08)
        
    username = customtkinter.CTkEntry(labelFrame, placeholder_text="Tên người dùng")
    username.place(relx=0.4,rely=0.03, relwidth=0.52, relheight=0.12)
        
    # User Address
    lb2 = Label(labelFrame,text="Địa chỉ : ", bg='#f5f0f0', fg='black')
    lb2.place(relx=0.05,rely=0.18, relheight=0.08)
  
    user_address = customtkinter.CTkEntry(labelFrame, placeholder_text="Địa chỉ người dùng")
    user_address.place(relx=0.4,rely=0.18, relwidth=0.52, relheight=0.12)

    # User Gender
    lb2 = Label(labelFrame,text="Giới tính(Nam/Nữ) : ", bg='#f5f0f0', fg='black')
    lb2.place(relx=0.05,rely=0.33, relheight=0.08)
  
    user_gender = customtkinter.CTkEntry(labelFrame, placeholder_text="Giới tính người dùng")
    user_gender.place(relx=0.4,rely=0.33, relwidth=0.52, relheight=0.12)

    # User email
    lb2 = Label(labelFrame,text="Email: ", bg='#f5f0f0', fg='black')
    lb2.place(relx=0.05,rely=0.48, relheight=0.08)
  
    user_email = customtkinter.CTkEntry(labelFrame, placeholder_text="Email người dùng")
    user_email.place(relx=0.4,rely=0.48, relwidth=0.52, relheight=0.123)

    lb2 = Label(labelFrame,text="Số điện thoại: ", bg='#f5f0f0', fg='black')
    lb2.place(relx=0.05,rely=0.63, relheight=0.08)
    user_phoneNumber = customtkinter.CTkEntry(labelFrame, placeholder_text="Số điện thoại người dùng")
    user_phoneNumber.place(relx=0.4,rely=0.63, relwidth=0.52, relheight=0.123)
    # Get hand info
    lb3 = Label(labelFrame,text="Lấy thông tin bàn tay:", bg='#f5f0f0', fg='black')
    lb3.place(relx=0.05,rely=0.85, relheight=0.08)

    # Get right hand button
    right_hand_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/right_hand.png").resize((30, 30)))
    getRightHandBtn = customtkinter.CTkButton(
        master=root, 
        image=right_hand_image, 
        text="RH1", 
        width=200, 
        height=30, 
        border_width=3, 
        corner_radius=10, 
        compound="right", 
        fg_color=("gray84", "blue"), 
        hover_color="#4486bd",
        command=get_right_hand_info1 
        )
    getRightHandBtn.place(relx=0.6,rely=0.52, relwidth=0.08, relheight=0.06)

    getRightHandBtn = customtkinter.CTkButton(
        master=root, 
        image=right_hand_image, 
        text="RH2", 
        width=200, 
        height=30, 
        border_width=3, 
        corner_radius=10, 
        compound="right", 
        fg_color=("gray84", "blue"), 
        hover_color="#4486bd",
        command=get_right_hand_info2
        )
    getRightHandBtn.place(relx=0.7,rely=0.52, relwidth=0.08, relheight=0.06)

    getRightHandBtn = customtkinter.CTkButton(
        master=root, 
        image=right_hand_image, 
        text="RH3", 
        width=200, 
        height=30, 
        border_width=3, 
        corner_radius=10, 
        compound="right", 
        fg_color=("gray84", "blue"), 
        hover_color="#4486bd",
        command=get_right_hand_info3 
        )
    getRightHandBtn.place(relx=0.8,rely=0.52, relwidth=0.08, relheight=0.06)

    # Get left hand button
    left_hand_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/left_hand.png").resize((30, 30)))
    getLeftHandBtn = customtkinter.CTkButton(
        master=root, 
        image=left_hand_image, 
        text="LH1", 
        width=200, 
        height=30, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "blue"), 
        hover_color="#4486bd",
        command=get_left_hand_info1 
        )
    getLeftHandBtn.place(relx=0.27,rely=0.52, relwidth=0.08, relheight=0.06)

    getLeftHandBtn = customtkinter.CTkButton(
        master=root, 
        image=left_hand_image, 
        text="LH2", 
        width=200, 
        height=30, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "blue"), 
        hover_color="#4486bd",
        command=get_left_hand_info2 
        )
    getLeftHandBtn.place(relx=0.37,rely=0.52, relwidth=0.08, relheight=0.06)

    getLeftHandBtn = customtkinter.CTkButton(
        master=root, 
        image=left_hand_image, 
        text="LH3", 
        width=200, 
        height=30, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "blue"), 
        hover_color="#4486bd",
        command=get_left_hand_info3 
        )
    getLeftHandBtn.place(relx=0.47,rely=0.52, relwidth=0.08, relheight=0.06)


    #Submit Button
    submit_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/submit.png").resize((20, 20)))
    SubmitBtn = customtkinter.CTkButton(
        master=root, 
        image=submit_image, 
        text="Thêm người dùng", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="right", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd",
        command=userRegister 
        )

    SubmitBtn.place(relx=0.28,rely=0.9, relwidth=0.18,relheight=0.08)
    
    quit_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/quit.png").resize((20, 20)))
    QuitBtn = customtkinter.CTkButton(
        master=root, 
        image=quit_image, 
        text="Thoát", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="right", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd",
        command=root.destroy 
        )
    QuitBtn.place(relx=0.53,rely=0.9, relwidth=0.15,relheight=0.08)
    
    root.mainloop()

if __name__ == "__main__":
    addUser()
