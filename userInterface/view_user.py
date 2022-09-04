from tkinter import *
from tkinter import ttk
import tkinter
from PIL import ImageTk,Image
import sys
sys.path.insert(1, "D:/Workspace/DATN/code")
from utils.utils import get_collection
import customtkinter
import PIL.Image, PIL.ImageTk

def View(): 
    # Creat Windows
    root = Toplevel()
    root.title("Hệ thống chứng thực người")
    root.geometry("960x540")
    root.resizable(width=False, height=False)
    
    my_col = get_collection("users")

    # Heading    
    headingFrame1 = Frame(root, bg="#FFBB00", bd=5)
    headingFrame1.place(relx=0.25, rely=0.1, relwidth=0.5, relheight=0.13)
        
    headingLabel = Label(headingFrame1, text="Kiểm tra thông tin người dùng", bg='#f5f0f0', fg='black', font=('Courier',15))
    headingLabel.place(relx=0, rely=0, relwidth=1, relheight=1)
    
    # Create Treeview
    labelFrame = Frame(root, bg='#f5f0f0')
    labelFrame.place(relx=0.1, rely=0.3, relwidth=0.8, relheight=0.5)
 
    columns = ("name", "gender", "address", "email", "phoneNumber")
    my_tree = ttk.Treeview(labelFrame, cursor="hand2")
    my_scrollbar = ttk.Scrollbar(labelFrame)
    my_scrollbar.pack(side='right', fill=Y)
    my_scrollbar.config(command=my_tree.yview)
    my_tree.config(xscrollcommand=my_scrollbar.set)


    my_tree["columns"] = columns
    id = 1
    my_tree.column("#0", width=50, minwidth=25)
    my_tree.column("name", width=160, anchor=W)
    my_tree.column("gender", width=60, anchor=W)
    my_tree.column("address", width=200, anchor=W)
    my_tree.column("email", width=200, anchor=W)
    my_tree.column("phoneNumber", width=80, anchor=W)

    my_tree.heading("#0", text="", anchor=W)
    my_tree.heading("name", text="Tên", anchor=W)
    my_tree.heading("gender", text="Giới tính", anchor=W)
    my_tree.heading("address", text="Địa chỉ", anchor=W)
    my_tree.heading("email", text="Email", anchor=W)
    my_tree.heading("phoneNumber", text="Số điện thoại", anchor=W)

    for user in my_col.find({}):
        my_tree.insert(
            parent="",
            index="end",
            text=id,
            values=(user["name"], user["gender"], user["address"], user["email"], user["phoneNumber"])
        )
        id += 1

    my_tree.pack(pady=20)

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
    QuitBtn.place(relx=0.4,rely=0.9, relwidth=0.18,relheight=0.08)
    
    root.mainloop()


if __name__ == '__main__':
    View()
