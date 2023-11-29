import os
from pathlib import Path
import streamlit as st

from FundusImage import FundusImage
from RoPModel import RoPModel
import constants

def view_handler(choice):
    st.subheader(choice.split(".")[0])

    model_path = constants.MODEL_H5_FOLDER + choice

    # needs further optimizations
    ropmodel = RoPModel(model_path, constants.MODEL_H5_FILES_AND_LABELS[choice])


    uploaded_file = st.file_uploader("Upload image", type=["png"])

    if uploaded_file:
        fundusImage = FundusImage(uploaded_file)

        og_img = fundusImage.get_original_image()
        eh_img = fundusImage.get_enhanced_image()

        col1, col2 = st.columns((1, 1))

        with col1:
            enhance = st.toggle("Enhance image?")
        with col2:
            predict = st.button("Predict")

        col3, col4 = st.columns((1, 1))

        with col3:
            st.image(og_img)
        with col4:
            if enhance:
                st.image(eh_img)

        if predict:
            if enhance:
                label, prob = ropmodel.predict_image(eh_img)
            else:
                label, prob = ropmodel.predict_image(og_img)

            st.text(label)
            st.text(prob)



def main():
    st.title(constants.APP_TITLE)

    available_models = list(constants.MODEL_H5_FILES_AND_LABELS.keys())


    choice = st.sidebar.selectbox("Choose model", available_models)

    if choice == available_models[0]:
        view_handler(available_models[0])

    elif choice == available_models[1]:
        view_handler(available_models[1])

    elif choice == available_models[2]:
        view_handler(available_models[2])

    elif choice == available_models[3]:
        view_handler(available_models[3])

    elif choice == available_models[4]:
        view_handler(available_models[4])

    else:
        st.subheader("This page should not be accessible")



if __name__ == "__main__":
    os.chdir(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
    main()