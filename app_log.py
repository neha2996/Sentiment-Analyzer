    
#Importing Libraries:
import streamlit as st 
import pickle
from PIL import Image
import re
import time
import numpy as np
from functions import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#-------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="App-Streamlit",page_icon="random",layout="wide",
                       menu_items={'Get Help': 'http://www.quickmeme.com/img/54/547621773e22705fcfa0e73bc86c76a05d4c0b33040fcb048375dfe9167d8ffc.jpg',
                                   'Report a bug': "https://w7.pngwing.com/pngs/839/902/png-transparent-ladybird-ladybird-bug-miscellaneous-presentation-insects-thumbnail.png",
                                   'About': "# This is a Sentiment Analyzer App based on Amazon reviews built with Streamlit. Log Regression model is used to train"})

@st.cache(allow_output_mutation=True) #For Autoupdate in app.


def loading_model():
    loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    return loaded_model, loaded_vec
with st.spinner('Model is being loaded..'):
    model, vec = loading_model() 
    

#-------------------------------------------------------------------------------------------------------------------------------
st.write("""
         # SENTIMENT ANALYZER
         """)

def processing(review):
    final_review = normalize_and_lemmaize(review) # Cleaned text to predict
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vec.fit_transform(np.array([final_review]))) # Converting to vectors
    return tfidf

def prediction(processed):
    pred=model.predict(processed) 
    return pred
#-------------------------------------------------------------------------------------------------------------------------------


# Decor Func:    
def decor():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Sentiment Analyzer App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    image = Image.open('sentiment-analysis.jpg')
    st.image(image, caption='')    

#--------------------------------------------------------------------------------------------------------------------------------

def main():

    decor()

    col1,col2=st.columns(2)
# taking Inputs:
    with col1:
        name=st.text_input("Customer's Name:")
        if name:
            st.write("# Customers Name: %s"%name)
        review = st.text_area('Input Review',placeholder="Type your review here")
        processed = processing(review)
        pred = prediction(processed)

        #---------------------------------------------------------------------------------------------------------------------------

    # Result:       
        result='Awaiting.. ....'
        if st.button("Predict"):
            with st.spinner('Wait for it...'):
                time.sleep(2)


            if pred==-1:
                result='This review is bad! Don not buy this product!!'
            elif pred==1:
                st.balloons()
                result= 'This review is good! Just buy this product!!'
                
        st.success(result)

    st.markdown("---")
    
#---------------------------------------------------------------------------------------------------------------------------

# Team Details:
    with col2:
        expander=st.expander("Team Details",expanded=False)
        with expander:
            st.write("Kasi")
            st.write("Neha")
            st.write("Anand")
            st.write("Hari")
            st.write("Nileena")
            st.write("Vishal")
#-----------------------------------------------------------------------------------------------------

# Program Starts:
if __name__=='__main__':
    main()
