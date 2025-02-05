import streamlit as st
from PIL import Image
import pytesseract
import openai
import json

# --- OpenAI API Evaluation Function ---
def evaluate_solution(api_key: str, extracted_text: str):
    """
    Calls the OpenAI API to evaluate the math solution.
    The prompt instructs the LLM to provide a score (out of 10) and constructive feedback.
    The expected output is a JSON object.
    """
    openai.api_key = api_key

    # Construct a prompt for the LLM.
    prompt = (
        "You are an expert math tutor and grader. Evaluate the following handwritten math solution "
        "for correctness, and provide a score out of 10 along with detailed, constructive feedback. "
        "If appropriate, suggest improvements. The solution is:\n\n"
        f"{extracted_text}\n\n"
        "Return your answer as a JSON object with keys: 'score' (an integer) and 'feedback' (a string)."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a math tutor and expert grader."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except Exception as e:
        return {"score": None, "feedback": f"OpenAI API call failed: {e}", "token_usage": {}}

    # Extract the response content and token usage.
    completion_text = response.choices[0].message.content.strip()
    token_usage = response.usage  # This is a dict with token counts.

    # Try to parse the completion text as JSON.
    try:
        evaluation = json.loads(completion_text)
        evaluation["token_usage"] = token_usage
    except Exception as e:
        # If parsing fails, return the raw text with token usage.
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
    # Open and display the uploaded image.
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Evaluate Solution"):
        # Check if the API key has been provided.
        if not api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        else:
            with st.spinner("Extracting text from image and evaluating solution..."):
                # --- Step 1: OCR Processing ---
                try:
                    extracted_text = pytesseract.image_to_string(image)
                except Exception as e:
                    st.error(f"Error during OCR processing: {e}")
                    extracted_text = ""

                if not extracted_text.strip():
                    st.error("No text could be extracted. Please ensure the image is clear and try again.")
                else:
                    st.subheader("Extracted Text")
                    st.text_area("OCR Output", value=extracted_text, height=200)

                    # --- Step 2: Evaluate the Extracted Text using OpenAI API ---
                    result = evaluate_solution(api_key, extracted_text)

                    st.subheader("Evaluation Results")
                    st.write(f"**Score:** {result.get('score', 'N/A')} / 10")
                    st.write(f"**Feedback:** {result.get('feedback', 'No feedback provided.')}")

                    # --- Step 3: Display Token Utilization ---
                    token_usage = result.get("token_usage", {})
                    if token_usage:
                        st.subheader("Token Utilization")
                        st.write(f"**Prompt Tokens:** {token_usage.get('prompt_tokens', 'N/A')}")
                        st.write(f"**Completion Tokens:** {token_usage.get('completion_tokens', 'N/A')}")
                        st.write(f"**Total Tokens:** {token_usage.get('total_tokens', 'N/A')}")
                    else:
                        st.info("No token usage information available.")

                    st.success("Evaluation complete!")
