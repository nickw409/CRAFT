from tkinter import *
import json
import requests


def submit_credentials(email, password):

    # validate credentials with firebase
    userJsonData = json.dumps({
        "email": email,
        "password": password,
        "returnSecureToken": True
    })

    response = requests.post("https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
                params={"key":"AIzaSyAv9i9m95XjsXAZPWYPZMJTlx_0u9GEHIA"},
                data=userJsonData)

    if "error" in response.json().keys():
        print("Invalid username and password")
    else:
        print("Login successful!")

def login():
    
    # create app and login frame
    app = Tk()
    app.geometry("400x400")
    app.title("Conveyor Belt Application")
    frame = Frame(app)


    # create labels and entry boxes
    title_label = Label(frame, text="Login").grid(columnspan=4, row=0, column=0,pady=2)
    email_label = Label(frame, text="Email:").grid(row=1, column=0, pady=2)
    email_entry = Entry(frame)
    email_entry.grid(columnspan=2, row=1, column=1,  padx=2, pady=2)
    pass_label = Label(frame, text="Password:").grid( row=2, column=0, pady=2)
    pass_entry = Entry(frame, show='*')
    pass_entry.grid(columnspan=2, row=2, column=1, padx=2, pady=2)

    # create button
    button = Button(frame, text="Submit", command=lambda:(submit_credentials(email_entry.get(), pass_entry.get())))
    button.grid(row=3, column=1, pady=5)
    button = Button(frame, text="Register")
    button.grid(row=3, column=2, pady=5) 

    frame.place(relx=.5, rely=.45,anchor= CENTER)
    app.mainloop()

login()