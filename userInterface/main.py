import sys
sys.path.insert(1, "D:/Workspace/DATN/code")

from tkinter import *
import PIL.Image, PIL.ImageTk
from add_user import *
from delete_user import *
from view_auth import *
from view_user import *
from auth import *
import customtkinter

def mainInterface(window):

    window.destroy()

    customtkinter.set_default_color_theme("dark-blue")
    image_size = 20
    root = Tk()
    root.title("Hệ thống chứng thực người")
    root.geometry("960x540")
    # root.resizable(0, 0)
    # root.state("zoomed")

    photo = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/bg.jpg").resize((1920, 1080)))
    bgPannel = Label(root, image=photo)
    bgPannel.image = photo
    bgPannel.pack(fill='both', expand='yes')

    navbarFrame = Frame(root,bg="#444444", bd=5)
    navbarFrame.place(x=0, y=0, relwidth=0.25, relheight=1)

    add_user_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/add.png").resize((image_size, image_size)))
    btn1 = customtkinter.CTkButton(
        master=navbarFrame, 
        image=add_user_image, 
        text="Thêm người dùng", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd", 
        command=addUser
    )
    btn1.place(relx=0.1, y=60)

    delete_user_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/delete.png").resize((image_size, image_size)))
    btn2 = customtkinter.CTkButton(
        master=navbarFrame, 
        image=delete_user_image, 
        text="Xóa người dùng", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd", 
        command=deleteUser
    )
    btn2.place(relx=0.1, y=110) 

    view_user_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/user.png").resize((image_size, image_size)))
    btn3 = customtkinter.CTkButton(
        master=navbarFrame, 
        image=view_user_image, 
        text="Kiểm tra người dùng", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd", 
        command=View
    )
    btn3.place(relx=0.1, y=160) 

    auth_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/hand.png").resize((image_size, image_size)))
    btn4 = customtkinter.CTkButton(
        master=navbarFrame, 
        image=auth_image, 
        text="Chứng thực người dùng", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd", 
        command=auth
    )
    btn4.place(relx=0.1, y=210) 

    btn5 = customtkinter.CTkButton(
        master=navbarFrame, 
        image=view_user_image, 
        text="Kiểm tra chứng thực", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd", 
        command=ViewAuth
    )
    btn5.place(relx=0.1, y=260)

    logout = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/logout.png").resize((image_size, image_size)))
    btn6 = customtkinter.CTkButton(
        master=navbarFrame, 
        image=logout, 
        text="Đăng xuất", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="right", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd", 
        command=root.destroy
    )

    btn6.place(relx=0.1, rely=0.9)       
    root.mainloop()

if __name__ == "__main__":
    window = Tk()
    mainInterface(window)
