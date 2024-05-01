import streamlit as st
import streamlit_authenticator as stauth
import ingredient_recognition

st.set_page_config(layout="wide")

names = ['LiangyuWang', 'Yamakata', 'administrator']
usernames = ['l_wang', 'yamakata', 'admin']
passwords = ['12345', '54321', 'ad1234']

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

    ingredient_recognition.main()
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
