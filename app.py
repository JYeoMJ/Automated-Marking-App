# Preprocessing of images:
# perspective corrrection, binarisation, segmentation
# Newest update: Using Handwritten Input via Canvas, powered by streamlit-drawable-canvas
# to add to docker
# same preprocesssing but can remove perspective correction. 

# im2latex Conversion:
# Each segmented region is passed to the im2latex_convert() placeholder, which in a real implementation would use a trained image-to-LaTeX model. 
# The outputs are concatenated to produce the full LaTeX code.
# Replace the placeholder function im2latex_convert() with your actual im2latex model integration to perform direct image-to-LaTeX conversion.

# LaTeX Conversion:
# The convert_to_latex function is a placeholder that performs simple symbol substitutions. This can be extended with more sophisticated logic or integrated with specialized tools.

# User Review & Evaluation:
# The LaTeX conversion result is displayed in an editable text area. The user can modify the result before submitting it for checkpoint scoring via the OpenAI API evaluation function.

# Submission & Checkpoint Scoring:
# When the user clicks the Submit LaTeX for Checkpoint Scoring button in the form, the (possibly edited) LaTeX is sent to the OpenAI API for evaluation. The result—including score, feedback, and token usage—is then displayed.

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import openai
import json
import logging
import os

# Set up logging for debugging.
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.2
# Optionally, set your OpenAI API key as an environment variable:
# openai.api_key = os.getenv("OPENAI_API_KEY")

# --- (Optional) Perspective Correction ---
def order_points(pts: np.array) -> np.array:
    """
    Orders points in the following order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

# not impt if using handwriting
# already removed from the actual processing at the bottom
def perspective_correction_cv2(image: np.array) -> np.array:
    """
    Applies perspective correction by detecting the largest 4-point contour.
    For canvas input, this step might be optional.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        logging.warning("No 4-point contour found for perspective correction.")
        return image

    pts = screenCnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- Binarization ---
def binarize_image_cv2(image: np.array) -> np.array:
    """
    Converts the image to grayscale and applies adaptive thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary

# --- Segmentation ---
def segment_image_cv2(image: np.array) -> list:
    """
    Segments the image into regions likely containing separate equations.
    Returns a list of image segments.
    """
    # Invert the image (assuming dark strokes on a light background)
    inverted = cv2.bitwise_not(image)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Filter out small noise
            segment = image[y:y+h, x:x+w]
            segments.append(segment)
    segments = sorted(segments, key=lambda seg: cv2.boundingRect(seg)[1])
    if not segments:
        segments = [image]
    return segments

# --- im2latex Conversion (Placeholder) ---
def im2latex_convert(segment: np.array) -> str:
    """
    Placeholder function for converting an image segment to LaTeX using an im2latex model.
    Replace this with your actual model inference code.
    """
    logging.info("Converting image segment to LaTeX using im2latex model.")
    # In a real implementation, preprocess the segment, run the model, and decode its output.
    return "Dummy LaTeX code for the equation segment."

def convert_image_to_latex(image: Image.Image) -> str:
    """
    Full pipeline: converts a PIL image to LaTeX code by applying (optional) perspective correction,
    binarization, segmentation, and then using im2latex to convert each segment.
    """
    # Convert PIL image to OpenCV BGR format.
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Optional: Perspective correction (for non-flat inputs).
    # corrected = perspective_correction_cv2(image_cv)
    
    # Binarize the image.
    binary = binarize_image_cv2(image_cv)
    
    # Segment the image.
    segments = segment_image_cv2(binary)
    
    # Convert each segment to LaTeX.
    latex_segments = [im2latex_convert(seg) for seg in segments]
    full_latex = "\n\n".join(latex_segments)
    return full_latex

# --- OpenAI API Evaluation Function ---
def evaluate_solution(api_key: str, solution_text: str):
    """
    Calls the OpenAI API to evaluate the math solution.
    The prompt instructs the LLM to return a score out of 10 and detailed feedback in JSON.
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

    completion_text = response.choices[0].message.content.strip()
    token_usage = response.usage

    try:
        evaluation = json.loads(completion_text)
        evaluation["token_usage"] = token_usage
    except Exception as e:
        logging.error(f"JSON parsing error: {e}")
        evaluation = {"score": None, "feedback": completion_text, "token_usage": token_usage}

    return evaluation

# --- Streamlit App Interface ---
st.title("AI-Powered Math Marking Assistant")
st.write("Handwrite your math solution in the digital canvas below.")

# Sidebar: Enter your OpenAI API key.
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Create a drawing canvas.
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",       # Transparent fill
    stroke_width=3,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=400,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

# Process the drawing when the user clicks the button.
if st.button("Convert Drawing to LaTeX"):
    if canvas_result.image_data is None:
        st.error("No drawing detected. Please write your solution on the canvas.")
    else:
        with st.spinner("Processing drawing..."):
            # Convert the canvas NumPy image (with RGBA channels) to a PIL image.
            # Remove the alpha channel if present.
            canvas_image = canvas_result.image_data
            if canvas_image.shape[2] == 4:
                canvas_image = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2RGB)
            pil_image = Image.fromarray(canvas_image)
            
            # Run the image-to-LaTeX conversion pipeline.
            latex_output = convert_image_to_latex(pil_image)
        
        st.subheader("LaTeX Conversion")
        # Allow user to review and edit the LaTeX code.
        with st.form(key="latex_form"):
            user_latex = st.text_area("LaTeX Output (editable)", value=latex_output, height=200)
            submit_checkpoint = st.form_submit_button("Submit LaTeX for Checkpoint Scoring")
            if submit_checkpoint:
                if not api_key:
                    st.error("Please enter your OpenAI API Key in the sidebar.")
                else:
                    with st.spinner("Evaluating solution..."):
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
