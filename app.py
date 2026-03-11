import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Medicinal Plant Classifier", layout="centered")

st.title("Medicinal Plant Leaf Classifier")
st.write("Upload a leaf image and press Predict.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            )
            response.raise_for_status()
            data = response.json()

            st.success("Prediction complete")
            st.write(f"**Predicted plant:** {data['plant_name']}")
            st.write(f"**Confidence:** {data['confidence']:.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
