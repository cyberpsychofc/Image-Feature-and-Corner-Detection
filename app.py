import streamlit as st
import cv2
import numpy as np
from PIL import Image
from groq import Groq

API_KEY = st.secrets["general"]["API_KEY"]

client = Groq(api_key=API_KEY)

def load_image(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return None

def sift_feature_matching(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches, kp1, kp2, good_matches

def orb_feature_matching(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

def harris_corner_detection(image):
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = 255
    return image

def shi_tomasi_corner_detection(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    else:
        gray = image  
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = corners.astype(np.int64)  # Fix for deprecated np.int0

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw corners

    return image

def response_generator(ALGORITHM):
    for chunk in client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Tell me something about {ALGORITHM} under 200 words. Your response should not include your chain of thought."
            }
        ],
        model="llama3-8b-8192",
        stream=True,
        temperature=0.5):

        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            full_response = ""
            choice = chunk.choices[0]  
            if hasattr(choice, "delta") and getattr(choice.delta, "content", None):
                content = choice.delta.content
                full_response += content  
                yield full_response

st.set_page_config(
    page_title="IFCD",
    layout="centered")

footer = f"""<footer style="padding-top: 200px;
                            font-size: small;
                            text-align: center;">
                Developed by Om Aryan</footer>"""

st.title("Image Feature & Corner Detection")

sift, orb, harris_corner, shi_tomasi = st.tabs(["SIFT", "ORB", "Harris Corner", "Shi-Tomasi"])

with sift:
    st.subheader("Scale-Invariant Feature Transform (SIFT) Feature Matching")
    st.write_stream(response_generator("Scale-Invariant Feature Transform (SIFT) Feature Matching"))
    uploaded_file1 = st.file_uploader("Upload first image of the object", type=["jpg", "png", "jpeg"])
    uploaded_file2 = st.file_uploader("Upload second image of the object", type=["jpg", "png", "jpeg"])

    if uploaded_file1 and uploaded_file2:
        img1 = load_image(uploaded_file1)
        img2 = load_image(uploaded_file2)
        img_matches, kp1, kp2, matches = sift_feature_matching(img1, img2)
        st.image(img_matches, caption="SIFT Matching", use_container_width=True)
                    
with orb:
    st.subheader("Oriented FAST and Rotated BRIEF (ORB) Feature Matching")
    st.write_stream(response_generator("Oriented FAST and Rotated BRIEF (ORB) Feature Matching"))
    uploaded_file1 = st.file_uploader("Upload first image of the object", type=["jpg", "png", "jpeg"], key="orb_file1")
    uploaded_file2 = st.file_uploader("Upload second image of the object", type=["jpg", "png", "jpeg"], key="orb_file2")

    if uploaded_file1 and uploaded_file2:
        img_1 = load_image(uploaded_file1)
        img_2 = load_image(uploaded_file2)
        orb_matches = orb_feature_matching(img_1, img_2)
        st.image(orb_matches, caption="ORB Matching", use_container_width=True)
        
with harris_corner:
    st.subheader("Harris Corner Detection")
    st.write_stream(response_generator("Harris Corner Detection"))
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    img = load_image(uploaded_file)
    if uploaded_file:
        harris_img = harris_corner_detection(img)
        st.image(harris_img, caption="Harris Corners", use_container_width=True)        
        

with shi_tomasi:
    st.subheader("Shi-Tomasi Corner Detection")
    st.write_stream(response_generator("Shi-Tomasi Corner Detection"))
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="shi_tomasi_file")
    img = load_image(uploaded_file)
    if uploaded_file:
        shi_tomasi_img = shi_tomasi_corner_detection(img)
        st.image(shi_tomasi_img, caption="Shi-Tomasi Corners", use_container_width=True)
        

st.markdown(footer, unsafe_allow_html=True)