from tkinter import *
import PIL.Image, PIL.ImageTk
from tkinter import messagebox

import sys
sys.path.insert(1, "D:/Workspace/DATN/code")

from utils.utils import get_collection
import customtkinter
 
def deleteUser(): 
    
    global userName, Canvas1, root
    root = Toplevel()
    root.title("Hệ thống chứng thực người")
    root.minsize(width=400,height=400)
    root.geometry("960x540")

    
    Canvas1 = Canvas(root)
    
    Canvas1.config(bg="#c6e2ff")
    Canvas1.pack(expand=True,fill=BOTH)
        
    headingFrame1 = Frame(root,bg="#FFBB00",bd=5)
    headingFrame1.place(relx=0.25,rely=0.1,relwidth=0.5,relheight=0.13)
        
    headingLabel = customtkinter.CTkLabel(headingFrame1, text="Xóa người dùng", bg_color='#f5f0f0', text_font=('Courier',15))
    headingLabel.place(relx=0,rely=0, relwidth=1, relheight=1)
    
    labelFrame = Frame(root,bg='#f5f0f0')
    labelFrame.place(relx=0.1,rely=0.3,relwidth=0.8,relheight=0.5)   
        
    # User name  to Delete
    lb2 = customtkinter.CTkLabel(labelFrame, text="Tên người dùng cần xóa:")
    lb2.place(relx=0.05,rely=0.5)
        
    userName = customtkinter.CTkEntry(labelFrame, width=120,placeholder_text="Nhập tên người dùng cần xóa")
    userName.place(relx=0.4,rely=0.5, relwidth=0.52)
    
    def deleteUser():
        name = userName.get()
        if len(name) != 0:
            my_col = get_collection("users")
            myQuery = {"name": name}
            user = my_col.find_one(myQuery)
            if user is not None:
                my_col.delete_one(myQuery)
                messagebox.showinfo("Thành công","Xóa người dùng thành công")
            else:
                messagebox.showwarning("Lỗi","Tên người dùng chưa chính xác. VUi lòng nhập chính xác tên người dùng")

        else:
            messagebox.showwarning("Lỗi","Nhập tên người dùng cần xóa")

    #Submit Button
    submit_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/submit.png").resize((20, 20)))
    SubmitBtn = customtkinter.CTkButton(
        master=root, 
        image=submit_image, 
        text="Xóa người dùng", 
        width=200, 
        height=40, 
        border_width=3, 
        corner_radius=10, 
        compound="right", 
        fg_color=("gray84", "gray25"), 
        hover_color="#4486bd",
        command=deleteUser 
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
    QuitBtn.place(relx=0.53,rely=0.9, relwidth=0.18,relheight=0.08)
    
    root.mainloop()

if __name__ == "__main__":
    deleteUser()
