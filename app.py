import streamlit as st
import cv2
import numpy as np
from PIL import Image
from groq import Groq
import tempfile
import base64

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

def response_generator(ALGORITHM, IMAGE_DATA):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image_path = tmp_file.name
        cv2.imwrite(image_path, IMAGE_DATA)  

    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{encoded_string}"

    for chunk in client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Explain what features, {ALGORITHM} (without any breifing about the algorithm) detects in this particular image given to you."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }
            ]
        }],
        model="llama-3.2-90b-vision-preview",
        stream=True,
        temperature=0.5,
        max_completion_tokens=1024,
        stop=None,
        top_p=1,):

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
    st.write("""
Scale-Invariant Feature Transform (SIFT) is a feature detection and description algorithm used in computer vision to match features between images. SIFT was developed by David Lowe in 2004 and has since become a widely used and popular algorithm in various applications such as object recognition, image retrieval, and tracking.
SIFT works by first detecting keypoints in an image, which are points with high intensity gradients or corners. These keypoints are then described by a 128-dimensional vector called a SIFT descriptor. The descriptor is computed by comparing the gradient magnitude and orientation at each keypoint to its neighbors, and then normalizing the results.
The SIFT descriptors are then matched between images by comparing the descriptors of keypoints in one image to the descriptors of keypoints in another image. The best match is found by calculating the Euclidean distance between the descriptors and selecting the one with the smallest distance. This process is repeated for all keypoints in both images, resulting in a set of matched keypoints.
""")
    uploaded_file1 = st.file_uploader("Upload first image of the object", type=["jpg", "png", "jpeg"])
    uploaded_file2 = st.file_uploader("Upload second image of the object", type=["jpg", "png", "jpeg"])

    if uploaded_file1 and uploaded_file2:
        img1 = load_image(uploaded_file1)
        img2 = load_image(uploaded_file2)
        img_matches, kp1, kp2, matches = sift_feature_matching(img1, img2)
        st.write_stream(response_generator("SIFT", img_matches))
        st.image(img_matches, caption="SIFT Matching", use_container_width=True)
        st.write(f"Number of keypoints in image 1: <code>{len(kp1)}</code>",unsafe_allow_html=True)
        st.write(f"Number of keypoints in image 2: <code>{len(kp2)}</code>",unsafe_allow_html=True)
        st.write(f"Number of matches: <code>{len(matches)}</code>",unsafe_allow_html=True)        
                    
with orb:
    st.subheader("Oriented FAST and Rotated BRIEF (ORB) Feature Matching")
    st.write("""
Oriented FAST and Rotated BRIEF (ORB) is a feature matching algorithm used in computer vision for object recognition and tracking. It is an extension of the FAST (Features from Accelerated Segment Test) algorithm, which detects key points in an image based on the intensity values of neighboring pixels.
The ORB algorithm adds rotation invariance to the FAST algorithm by using a rotation invariant descriptor, known as BRIEF (Binary Robust Independent Elementary Features). The BRIEF descriptor is a binary string that describes the local structure of the image around a key point. The descriptor is created by comparing the intensity values of neighboring pixels in a specific pattern.
The ORB algorithm uses the FAST algorithm to detect key points in an image and then computes the BRIEF descriptor for each key point. The descriptors are then matched between two images to find corresponding key points, allowing for object recognition and tracking. ORB is a fast and robust feature matching algorithm, making it suitable for real-time applications.
""")
    uploaded_file1 = st.file_uploader("Upload first image of the object", type=["jpg", "png", "jpeg"], key="orb_file1")
    uploaded_file2 = st.file_uploader("Upload second image of the object", type=["jpg", "png", "jpeg"], key="orb_file2")

    if uploaded_file1 and uploaded_file2:
        img_1 = load_image(uploaded_file1)
        img_2 = load_image(uploaded_file2)
        orb_matches = orb_feature_matching(img_1, img_2)
        st.write_stream(response_generator("ORB Matching", orb_matches))
        st.image(orb_matches, caption="ORB Matching", use_container_width=True)
        
with harris_corner:
    st.subheader("Harris Corner Detection")
    st.write("""
Harris Corner Detection is a feature detection algorithm used in computer vision to identify corners in digital images. It is a widely used and effective method for detecting corners, which are points in an image where the gradient magnitude is high and the gradient direction changes significantly. The algorithm was developed by Chris Harris and Mike Stephens in 1988.
The Harris Corner Detection algorithm works by analyzing the intensity values of neighboring pixels in an image. It calculates the autocorrelation matrix of the pixel values, which represents the relationship between the pixel values and their neighbors. The algorithm then computes the determinant and trace of the autocorrelation matrix, which are used to calculate the corner response function. The corners are detected by finding the local maxima of the corner response function.
Harris Corner Detection is known for its simplicity, robustness, and accuracy. It is widely used in various computer vision applications, including object recognition, tracking, and image registration.
""")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    img = load_image(uploaded_file)
    if uploaded_file:
        harris_img = harris_corner_detection(img)
        st.write_stream(response_generator("Harris Corner Detection", harris_img))
        st.image(harris_img, caption="Harris Corners", use_container_width=True)        
        

with shi_tomasi:
    st.subheader("Shi-Tomasi Corner Detection")
    st.write("""
Shi-Tomasi Corner Detection is a widely used corner detection algorithm in computer vision. It was introduced by Shi and Tomasi in 1994 and is based on the Harris corner detector. The algorithm works by finding the points in an image where the intensity changes significantly in all directions, which are typically the corners.
The Shi-Tomasi algorithm uses a similar approach to the Harris detector, but it is more efficient and robust. It uses a threshold value to determine whether a point is a corner or not. The algorithm first calculates the autocorrelation matrix of the image, which represents the intensity values of the image at different points. It then calculates the eigenvalues of the matrix, which represent the intensity changes in the image.
The algorithm selects the points with the largest eigenvalues as corners, as these are the points where the intensity changes the most. The algorithm also uses a non-maximum suppression technique to remove the false corners and refine the detected corners. The Shi-Tomasi algorithm is widely used in various computer vision applications, including object recognition, tracking, and stereo matching.
""")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="shi_tomasi_file")
    img = load_image(uploaded_file)
    if uploaded_file:
        shi_tomasi_img = shi_tomasi_corner_detection(img)
        st.write_stream(response_generator("Shi-Tomasi Corner Detection", shi_tomasi_img))
        st.image(shi_tomasi_img, caption="Shi-Tomasi Corners", use_container_width=True)
        

st.markdown(footer, unsafe_allow_html=True)