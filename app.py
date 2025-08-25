import os
from pathlib import Path
import pandas as pd
import streamlit as st
st.set_page_config(page_title='Photo finder',layout='wide')
from dotenv import dotenv_values
from openai import OpenAI
import base64

env = dotenv_values(".env")

# OpenAI API key protection

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Podaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

def prepare_image_for_open_ai(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:image/png;base64,{image_data}"

def describe_image(image_path):
    openai_client=get_openai_client()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Stwórz opis obrazka, jakie widzisz tam elementy?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": prepare_image_for_open_ai(image_path),
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content

DATA_PATH=Path('data')
RAW_PATH=Path('data','raw')

st.title(":camera: Znajdywacz zdjęć")
photos=st.file_uploader(
 "Dodaj zdjęcie",
 accept_multiple_files=True
 )

if photos:
 os.makedirs(DATA_PATH,exist_ok=True)
 for file in photos:
  time=str(pd.Timestamp('now'))
  time=time.replace(' ','_')
  time=time.replace(':','-')
  time=time.replace('.',',')
  os.makedirs(DATA_PATH/time)
  with open(DATA_PATH/time/file.name,'wb') as f:
   f.write(file.read())
  with open(f'{DATA_PATH/time}/opis.txt','w') as f:
   image_description=describe_image(DATA_PATH/time/file.name)
   openai_client=get_openai_client()
   result=openai_client.embeddings.create(
     input=[image_description],
     model='text-embedding-3-large'
   )
   result[time]=image_description
   f.write(image_description)
 photos=False
 st.markdown("""
            <script>
            window.location.reload();
            window.location.reload();
            </script>
             """,
              unsafe_allow_html=True 
            )
 st.success(f"Zapisano pliki w {DATA_PATH}")
