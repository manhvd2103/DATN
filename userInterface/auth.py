from tkinter import *
import tkinter.filedialog
from tkinter import messagebox
from datetime import datetime
import torch
import os
import sys
sys.path.insert(1, "D:/Workspace/DATN/code")
import time
import pickle
from utils.utils import embedding_extraction, get_collection
from utils.transforms import test_transforms
from configs.config import DEVICE
import PIL.Image, PIL.ImageTk
import customtkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

canvas = None

def auth_process():
    global canvas
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.tiff .jpg .png')])
    if path is None:
        return

    cos = torch.nn.CosineSimilarity()
    start = time.time()
    query, fig = embedding_extraction(
                image_path=os.path.join(path),
                device=DEVICE,
                transforms=test_transforms,
                draw=True,
                name=""
    )
    canvas = FigureCanvasTkAgg(fig, left_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()
    user_collection = get_collection("users")
    authentication_collection = get_collection("auth")
    query = torch.from_numpy(query[0])
    cosine_values = []
    for user in user_collection.find({}, {"name":1, "email": 1, "phoneNumber": 1, "address": 1, "left_hand1": 1, "left_hand2": 1, "left_hand3": 1,"right_hand1": 1, "right_hand2": 1, "right_hand3": 1}):
        left_hand1_info_numpy = pickle.loads(user.get("left_hand1"))
        left_hand1_info_tensor = torch.from_numpy(left_hand1_info_numpy)
        left_cosine_value1 = cos(query, left_hand1_info_tensor)
        cosine_values.append(left_cosine_value1)

        left_hand2_info_numpy = pickle.loads(user.get("left_hand2"))
        left_hand2_info_tensor = torch.from_numpy(left_hand2_info_numpy)
        left_cosine_value2 = cos(query, left_hand2_info_tensor)
        cosine_values.append(left_cosine_value2)

        left_hand3_info_numpy = pickle.loads(user.get("left_hand3"))
        left_hand3_info_tensor = torch.from_numpy(left_hand3_info_numpy)
        left_cosine_value3 = cos(query, left_hand3_info_tensor)
        cosine_values.append(left_cosine_value3)

        right_hand_info1_numpy = pickle.loads(user.get("right_hand1"))
        right_hand_info1_tensor = torch.from_numpy(right_hand_info1_numpy)
        right_cosine_value1 = cos(query, right_hand_info1_tensor)
        cosine_values.append(right_cosine_value1)

        right_hand_info2_numpy = pickle.loads(user.get("right_hand2"))
        right_hand_info2_tensor = torch.from_numpy(right_hand_info2_numpy)
        right_cosine_value2 = cos(query, right_hand_info2_tensor)
        cosine_values.append(right_cosine_value2)

        right_hand_info3_numpy = pickle.loads(user.get("right_hand3"))
        right_hand_info3_tensor = torch.from_numpy(right_hand_info3_numpy)
        right_cosine_value3 = cos(query, right_hand_info3_tensor)
        cosine_values.append(right_cosine_value3)

        similarity = 0

        if(left_cosine_value1.item() >= 0.8 or right_cosine_value1.item() >=0.8 or left_cosine_value2.item() >= 0.8 or right_cosine_value2.item() >=0.8 or left_cosine_value3.item() >= 0.8 or right_cosine_value3.item() >=0.8):
            for consine_value in cosine_values:
                if consine_value > similarity:
                    similarity = consine_value
            print(similarity)
            authentication_collection.insert_one({"user_name": user["name"], "email": user["email"], "address": user["address"], "phoneNumber": user["phoneNumber"] , "Time_auth": datetime.utcnow(), "Similarity": similarity.item()})
            print("Chứng thực hoàn tất với thời gian {}s".format(round(time.time() - start), 2))
            messagebox.showinfo("Thành công", "Người dùng: {} đã chứng thực thành công".format(user.get("name")))
            return
    print("Chứng thực hoàn tất với thời gian {}s".format(round(time.time() - start), 2))    
    messagebox.showinfo("Cảnh báo", "Người chứng thực không có trong cơ sở dữ liệu!")
    return
def auth(): 
    
    global userName, Canvas1, root, left_frame
    image_size = 50
    root = Toplevel()
    root.title("Hệ thống chứng thực người")
    root.minsize(width=400,height=400)
    root.geometry("960x540")

    
    Canvas1 = Canvas(root)
    
    Canvas1.config(bg="#c6e2ff")
    Canvas1.pack(expand=True,fill=BOTH)
        
    headingFrame1 = Frame(root,bg="#FFBB00",bd=5)
    headingFrame1.place(relx=0.25,rely=0.1,relwidth=0.5,relheight=0.13)
        
    headingLabel = Label(headingFrame1, text="Chứng thực người dùng", bg='#f5f0f0', fg='black', font=('Courier',15))
    headingLabel.place(relx=0,rely=0, relwidth=1, relheight=1)
    
    labelFrame = Frame(root,bg='#f5f0f0')
    labelFrame.place(relx=0.1,rely=0.3,relwidth=0.8,relheight=0.5)   
    left_frame = Frame(root, bg="white", bd=5)
    left_frame.place(relx=0.1,rely=0.5, relwidth=0.8, relheight= 0.35) 
    lb2 = Label(labelFrame,text="Chọn ảnh lòng bàn tay: ", bg='#f5f0f0', fg='black')
    lb2.place(relx=0.05,rely=0.2)
    
    # Authencation
    hand_image = PIL.ImageTk.PhotoImage(PIL.Image.open("../images/hand.png").resize((image_size, image_size)))
    getLeftHandBtn = customtkinter.CTkButton(
        master=root, 
        image=hand_image, 
        text="" ,
        width=200, 
        height=30, 
        border_width=3, 
        corner_radius=10, 
        compound="left", 
        fg_color=("gray84", "blue"), 
        hover_color="#4486bd",
        command=auth_process 
        )
    getLeftHandBtn.place(relx=0.6, rely=0.35, relwidth=0.18, relheight=0.12)
    
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

if __name__ == "__main__":
    auth()