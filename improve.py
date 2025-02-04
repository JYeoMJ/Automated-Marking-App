# Preprocessing:
# The preprocess_image function converts the image to grayscale and applies binary thresholding to help improve the OCR results.

# LaTeX Conversion:
# The convert_to_latex function is a placeholder that performs simple symbol substitutions. This can be extended with more sophisticated logic or integrated with specialized tools.

# Submission & Checkpoint Scoring:
# When the user clicks the Submit LaTeX for Checkpoint Scoring button in the form, the (possibly edited) LaTeX is sent to the OpenAI API for evaluation. The result—including score, feedback, and token usage—is then displayed.

import streamlit as st
from PIL import Image, ImageOps
import pytesseract
import openai
import json
import logging
import os

# Set up logging to help with debugging and error tracking.
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.2

# Optionally, load your OpenAI API key from an environment variable:
# openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Preprocessing Function ---
def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the image to improve OCR accuracy.
    Converts the image to grayscale and applies binary thresholding.
    """
    try:
        # Convert the image to grayscale.
        gray_image = image.convert('L')
        # Apply binary thresholding to enhance contrast.
        threshold = 128  # Adjust threshold value as needed.
        binary_image = gray_image.point(lambda x: 0 if x < threshold else 255, '1')
        logging.info("Image preprocessing successful.")
        return binary_image
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        # Return the original image if preprocessing fails.
        return image

# --- LaTeX Conversion Function ---
def convert_to_latex(ocr_text: str) -> str:
    """
    Convert OCR text of math expressions into LaTeX.
    This placeholder performs basic substitutions.
    Extend this function for robust conversion.
    """
    try:
        latex_text = ocr_text
        # Basic symbol replacements.
        latex_text = latex_text.replace("√", "\\sqrt{}")
        latex_text = latex_text.replace("∑", "\\sum")
        latex_text = latex_text.replace("∞", "\\infty")
        # Additional substitutions can be added here.
        logging.info("LaTeX conversion performed.")
        return latex_text
    except Exception as e:
        logging.error(f"Error during LaTeX conversion: {e}")
        return ocr_text

# --- OpenAI API Evaluation Function ---
def evaluate_solution(api_key: str, solution_text: str):
    """
    Calls the OpenAI API to evaluate the math solution.
    Constructs a prompt that instructs the LLM to provide a score out of 10 
    and detailed, constructive feedback. Expected output is a JSON object.
    """
    openai.api_key = api_key

    prompt = (
        "You are an expert math tutor and grader. Evaluate the following math solution "
        "for correctness, and provide a score out of 10 along with detailed, constructive feedback. "
        "If appropriate, suggest improvements. The solution (in LaTeX format) is:\n\n"
        f"{solution_text}\n\n"
        "Return your answer as a JSON object with keys: 'score' (an integer) and 'feedback' (a string)."
    )

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a math tutor and expert grader."},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
        )
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return {"score": None, "feedback": f"OpenAI API call failed: {e}", "token_usage": {}}

    # Extract response content and token usage.
    completion_text = response.choices[0].message.content.strip()
    token_usage = response.usage

    # Attempt to parse the completion text as JSON.
    try:
        evaluation = json.loads(completion_text)
        evaluation["token_usage"] = token_usage
    except Exception as e:
        logging.error(f"JSON parsing error: {e}")
        # Fallback: return the raw text with token usage.
        evaluation = {"score": None, "feedback": completion_text, "token_usage": token_usage}

    return evaluation

# --- Streamlit App Interface ---
st.title("AI-Powered Math Marking Assistant Prototype")
st.write("Upload an image of your handwritten math solution below.")

# Sidebar: Enter your OpenAI API key.
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# File uploader for the handwritten solution image.
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Error opening image: {e}")
        st.stop()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to process the image.
    if st.button("Process Image"):
        if not api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        else:
            with st.spinner("Processing image..."):
                # Step 1: Preprocess the image.
                preprocessed_image = preprocess_image(image)
                st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

                # Step 2: OCR Processing.
                try:
                    extracted_text = pytesseract.image_to_string(preprocessed_image)
                except Exception as e:
                    st.error(f"Error during OCR processing: {e}")
                    extracted_text = ""

                if not extracted_text.strip():
                    st.error("No text could be extracted. Please ensure the image is clear and try again.")
                else:
                    st.subheader("Extracted Text")
                    st.text_area("OCR Output", value=extracted_text, height=200)

                    # Step 3: LaTeX Conversion.
                    latex_output = convert_to_latex(extracted_text)
                    st.subheader("LaTeX Conversion")

                    # Use a form to let the user review and edit the LaTeX conversion.
                    with st.form(key="latex_form"):
                        user_latex = st.text_area("LaTeX Output (editable)", value=latex_output, height=200)
                        submit_checkpoint = st.form_submit_button("Submit LaTeX for Checkpoint Scoring")
                        if submit_checkpoint:
                            with st.spinner("Evaluating solution..."):
                                # Step 4: Evaluate the (possibly edited) LaTeX solution.
                                result = evaluate_solution(api_key, user_latex)
                                st.subheader("Evaluation Results")
                                st.write(f"**Score:** {result.get('score', 'N/A')} / 10")
                                st.write(f"**Feedback:** {result.get('feedback', 'No feedback provided.')}")
                                token_usage = result.get("token_usage", {})
                                if token_usage:
                                    st.subheader("Token Utilization")
                                    st.write(f"**Prompt Tokens:** {token_usage.get('prompt_tokens', 'N/A')}")
                                    st.write(f"**Completion Tokens:** {token_usage.get('completion_tokens', 'N/A')}")
                                    st.write(f"**Total Tokens:** {token_usage.get('total_tokens', 'N/A')}")
                                st.success("Evaluation complete!")
