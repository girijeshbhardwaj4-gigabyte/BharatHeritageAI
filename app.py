
import streamlit as st

from src.model import HeritageVQA
from src.image_processor import process_image


# Page configuration
st.set_page_config(
    page_title="BharatHeritage AI",
    page_icon="🏛️",
    layout="centered"
)


# Load the AI model only once
@st.cache_resource
def load_vqa_model():
    return HeritageVQA()


# Application heading
st.title("🏛️ BharatHeritage AI")

st.subheader(
    "An Intelligent Image-Based Question Answering System"
)

st.write(
    "Upload an image of an Indian heritage monument "
    "and ask questions about what you see."
)

st.divider()


# Image upload section
uploaded_file = st.file_uploader(
    "Upload a Heritage Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        # Process uploaded image
        image = process_image(uploaded_file)

        # Display image
        st.image(
            image,
            caption="Uploaded Heritage Image",
            use_container_width=True
        )

        st.divider()

        # Question input
        question = st.text_input(
            "Ask a Question About This Image",
            placeholder="Example: What is visible in this image?"
        )

        # Generate answer
        if st.button("Generate Answer", type="primary"):

            if not question.strip():

                st.warning("Please enter a question.")

            else:

                try:
                    with st.spinner("Analyzing image..."):

                        model = load_vqa_model()

                        answer = model.answer_question(
                            image,
                            question.strip()
                        )

                    st.subheader("AI Response")
                    st.success(answer)

                except Exception as error:

                    st.error(
                        f"Unable to generate an answer: {error}"
                    )

    except ValueError as error:

        st.error(str(error))