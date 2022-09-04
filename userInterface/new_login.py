import sys
sys.path.insert(1, "D:/Workspace/DATN/code")

from tkinter import *
from tkinter import messagebox
import PIL
from utils.utils import get_collection
from add_user import *
from delete_user import *
from view_user import *
from auth import *
from view_auth import *
import customtkinter
from main import *

def login():
    window = Tk()
    window.geometry('1902x1080')
    window.resizable(0, 0)
    window.state('zoomed')
    window.title('Login Page')

    # Background image
    bg_login_frame = PIL.Image.open("../images\\background1.png")
    photo = PIL.ImageTk.PhotoImage(bg_login_frame)
    bg_login_panel = Label(window, image=photo)
    bg_login_panel.image = photo
    bg_login_panel.pack(fill='both', expand='yes')
        
    # Login Frame
    lgn_frame = Frame(window, bg='#e7f0ea', width=950, height=600)
    lgn_frame.place(x=280, y=120)
    heading = Label(lgn_frame, justify=CENTER, text="HỆ THỐNG CHỨNG THỰC NGƯỜI", bd=5, relief=FLAT)
    heading.place(x=0, y=30, width=950, height=35)

        # Left Side Image
    login_side_image = PIL.Image.open('../images\\vector.png')
    photo = PIL.ImageTk.PhotoImage(login_side_image)
    login_side_image_label = Label(lgn_frame, image=photo, bg='#e7f0ea')
    login_side_image_label.image = photo
    login_side_image_label.place(x=5, y=100)

    # Sign In 
    sign_in_image = PIL.Image.open('../images\\hyy.png')
    photo = PIL.ImageTk.PhotoImage(sign_in_image)
    sign_in_image_label = Label(lgn_frame, image=photo, bg='#e7f0ea')
    sign_in_image_label.image = photo
    sign_in_image_label.place(x=620, y=130)
    sign_in_label = Label(lgn_frame, text="Đăng nhập", bg="#e7f0ea",fg="Black", font=("yu gothic ui", 17, "bold"))
    sign_in_label.place(x=633, y=240)

    # Username
    login_username_label = Label(lgn_frame, text="Email", bg="#e7f0ea", fg="black", font=("yu gothic ui", 13, "bold"))
    login_username_label.place(x=550, y=300)
    login_username_entry = Entry(lgn_frame, highlightthickness=0, relief=FLAT, bg="#e7f0ea", fg="Black", font=("yu gothic ui ", 12, "bold"))
    login_username_entry.place(x=580, y=335, width=270)
    username_line = Canvas(lgn_frame, width=300, height=2.0, bg="#bdb9b1", highlightthickness=0)
    username_line.place(x=550, y=359)
    username_icon = PIL.Image.open('../images\\username_icon.png')
    photo = PIL.ImageTk.PhotoImage(username_icon)
    username_icon_label = Label(lgn_frame, image=photo, bg='#e7f0ea')
    username_icon_label.image = photo
    username_icon_label.place(x=550, y=332)

    # Password
    password_label = Label(lgn_frame, text="Password", bg="#e7f0ea", fg="#4f4e4d", font=("yu gothic ui", 13, "bold"))
    password_label.place(x=550, y=380)

    password_entry = Entry(lgn_frame, highlightthickness=0, relief=FLAT, bg="#e7f0ea", fg="black", font=("yu gothic ui", 12, "bold"), show="*")
    password_entry.place(x=580, y=416, width=244)

    password_line = Canvas(lgn_frame, width=300, height=2.0, bg="#bdb9b1", highlightthickness=0)
    password_line.place(x=550, y=440)
    password_icon = PIL.Image.open('../images\\password_icon.png')
    photo = PIL.ImageTk.PhotoImage(password_icon)
    password_icon_label = Label(lgn_frame, image=photo, bg='#e7f0ea')
    password_icon_label.image = photo
    password_icon_label.place(x=550, y=414)

    def password_show():
        hide_button = Button(lgn_frame, image=hide_image, command=password_hide, relief=FLAT, activebackground="white", borderwidth=0, background="white", cursor="hand2")
        hide_button.place(x=860, y=420)
        password_entry.config(show='')

    
    def password_hide():
        show_button = Button(lgn_frame, image=show_image, command=password_show, relief=FLAT, activebackground="white", borderwidth=0, background="white", cursor="hand2")
        show_button.place(x=860, y=420)
        password_entry.config(show='*')

    # Show/hide password 
    show_image = PIL.ImageTk.PhotoImage(file='../images\\show.png')
    hide_image = PIL.ImageTk.PhotoImage(file='../images\\hide.png')
    show_button = Button(lgn_frame, image=show_image, command=password_show, relief=FLAT, activebackground="white", borderwidth=0, background="white", cursor="hand2")
    show_button.place(x=860, y=420)

    def authlogin():
        email = login_username_entry.get()
        password = password_entry.get()
        if len(email) != 0 and len(password) != 0:
            my_col = get_collection ("login")
            user = my_col.find_one({"email": email})
            if user != None:
                if user["password"] == password:
                    mainInterface(window)
                else:
                    messagebox.showwarning("", "Tên đăng nhập hoặc mật khẩu không đúng")
            else:
                messagebox.showwarning("", "Tên đăng nhập hoặc mật khẩu không đúng")
        else:
                messagebox.showwarning("", "Vui lòng điền đầy đủ Tên đăng nhập và Mật khẩu")

    # Login Button
    lgn_button = PIL.Image.open('../images\\btn1.png')
    photo = PIL.ImageTk.PhotoImage(lgn_button)
    lgn_button_label = Label(lgn_frame, image=photo, bg='#e7f0ea')
    lgn_button_label.image = photo
    lgn_button_label.place(x=580, y=450)
    login = Button(lgn_button_label, text='ĐĂNG NHẬP', font=("yu gothic ui", 13, "bold"), width=25, bd=0, bg='#3047ff', activebackground='#3047ff', fg='white', command=authlogin)
    login.place(x=20, y=10)
    login.pack()

    window.mainloop()
            
if __name__ == '__main__':
    login()