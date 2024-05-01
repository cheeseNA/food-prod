import streamlit as st
import streamlit_authenticator as stauth
import ingredient_recognition_jp
import json

st.set_page_config(
    page_title="RecipeLog2023",
    page_icon=":curry:",
    layout="wide", 
    initial_sidebar_state="collapsed"
    )

# with open('user_informations.json', 'r', encoding='utf-8') as file:
#     user_infos = json.load(file) 

# names = user_infos["names"]
# usernames = user_infos["usernames"]
# passwords = user_infos["passwords"]

names = ['LiangyuWang', 'Yamakata', 'administrator', 'f001', 'f002', 'f003', 'f004', 'f005', 'f006', 'f007', 'f008', 'f009', 'f010']
usernames = ['l_wang', 'yamakata', 'admin', 'f001', 'f002', 'f003', 'f004', 'f005', 'f006', 'f007', 'f008', 'f009', 'f010']
passwords = ['12345', '54321', 'ad1234', 'foodlog001', 'foodlog002', 'foodlog003', 'foodlog004', 'foodlog005', 'foodlog006', 'foodlog007', 'foodlog008', 'foodlog009', 'foodlog0010']

hashed_passwords = stauth.Hasher(passwords).generate()

credentials = {"usernames":{}}
for uname, name, pwd in zip(usernames, names, hashed_passwords):
    user_dict = {"name": name, "password": pwd}
    credentials["usernames"].update({uname: user_dict})

authenticator = stauth.Authenticate(credentials,
    'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    with st.container():
        cols1,cols2 = st.columns(2)
        cols1.write('Welcome *%s*' % (name))
        with cols2.container():
            authenticator.logout('Logout', 'main')

    ingredient_recognition_jp.main(username)
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
