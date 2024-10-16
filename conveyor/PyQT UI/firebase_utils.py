import json
import requests
from firebase_admin import auth, credentials
import firebase_admin

cred = credentials.Certificate("auth_key.json")
firebase_admin.initialize_app(cred)

# login user and save email
def login(email, password):

    # validate credentials with firebase
    userJsonData = json.dumps({
        "email": email,
        "password": password,
        "returnSecureToken": True
    })

    response = requests.post("https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
                params={"key":"AIzaSyAv9i9m95XjsXAZPWYPZMJTlx_0u9GEHIA"},
                data=userJsonData)

    if response.status_code == 200:
        return response.json().get('email')
    else:
        return False

# register a new user  
def register(email, password):
    try:
        auth.create_user(email=email, password=password)
        return True
    except Exception as error:
        return False
        