import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance

st.set_page_config(page_title="Interactive Image Editor", layout="wide")
st.title(" Interactive Image Editing App")


# Upload

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.stop()

# Read and convert to OpenCV format
pil_img = Image.open(uploaded_file)
img = cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR) # BGR for OpenCV
st.sidebar.header("Choose Editing Options")


# Basic Editing

st.sidebar.subheader("Basic Editing")
resize = st.sidebar.checkbox("Resize")
crop = st.sidebar.checkbox("Crop")
rotate = st.sidebar.checkbox("Rotate")
flip = st.sidebar.checkbox("Flip")
bright_contrast = st.sidebar.checkbox("Adjust Brightness & Contrast")

edited = img.copy()

if resize:
    width = st.sidebar.slider("Width", 50, img.shape[1]*2, img.shape[1])
    height = st.sidebar.slider("Height", 50, img.shape[0]*2, img.shape[0])
    edited = cv2.resize(edited, (width, height))

if crop:
    top = st.sidebar.number_input("Top", 0, edited.shape[0], 0)
    bottom = st.sidebar.number_input("Bottom", 0, edited.shape[0], 0)
    left = st.sidebar.number_input("Left", 0, edited.shape[1], 0)
    right = st.sidebar.number_input("Right", 0, edited.shape[1], 0)

    edited = edited[int(top):edited.shape[0]-int(bottom), int(left):edited.shape[1]-int(right)]

if rotate:
    angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
    (h, w) = edited.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    edited = cv2.warpAffine(edited, M, (w, h))

if flip:
    flip_choice = st.sidebar.radio("Flip Direction", ["Horizontal", "Vertical"])
    if flip_choice == "Horizontal":
        edited = cv2.flip(edited, 1)
    else:
        edited = cv2.flip(edited, 0)

if bright_contrast:
    brightness = st.sidebar.slider("Brightness", 0.0, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.0, 2.0, 1.0)
    pil_tmp = Image.fromarray(cv2.cvtColor(edited, cv2.COLOR_BGR2RGB))
    enhancer_b = ImageEnhance.Brightness(pil_tmp)
    bright_img = enhancer_b.enhance(brightness)
    enhancer_c = ImageEnhance.Contrast(bright_img)
    edited = cv2.cvtColor(np.array(enhancer_c.enhance(contrast)), cv2.COLOR_RGB2BGR)


# Colour Transformations

st.sidebar.subheader("Colour Transformations")
color_mode = st.sidebar.selectbox(
    "Convert colour space",
    ["None", "Grayscale", "HSV", "Lab"]
)
if color_mode == "Grayscale":
    edited = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
elif color_mode == "HSV":
    edited = cv2.cvtColor(edited, cv2.COLOR_BGR2HSV)
elif color_mode == "Lab":
    edited = cv2.cvtColor(edited, cv2.COLOR_BGR2Lab)


# Filtering & Enhancement

st.sidebar.subheader("Filtering & Enhancement")
filter_choice = st.sidebar.selectbox(
    "Apply filter",
    ["None", "Blur", "Gaussian Blur", "Sharpen", "Edge Detection (Canny)"]
)
if filter_choice == "Blur":
    k = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
    edited = cv2.blur(edited, (k, k))
elif filter_choice == "Gaussian Blur":
    k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
    edited = cv2.GaussianBlur(edited, (k, k), 0)
elif filter_choice == "Sharpen":
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    edited = cv2.filter2D(edited, -1, kernel)
elif filter_choice == "Edge Detection (Canny)":
    t1 = st.sidebar.slider("Threshold1", 0, 255, 100)
    t2 = st.sidebar.slider("Threshold2", 0, 255, 200)
    edited = cv2.Canny(edited, t1, t2)

# Noise Removal

st.sidebar.subheader("Noise Removal")
noise_choice = st.sidebar.selectbox(
    "Denoising",
    ["None", "Median Filter", "Bilateral Filter"]
)
if noise_choice == "Median Filter":
    k = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
    edited = cv2.medianBlur(edited, k)
elif noise_choice == "Bilateral Filter":
    d = st.sidebar.slider("Diameter", 1, 20, 9)
    sigmaColor = st.sidebar.slider("SigmaColor", 1, 150, 75)
    sigmaSpace = st.sidebar.slider("SigmaSpace", 1, 150, 75)
    edited = cv2.bilateralFilter(edited, d, sigmaColor, sigmaSpace)


# Display

st.subheader("Edited Image")
if len(edited.shape) == 2:  # grayscale
    st.image(edited, use_container_width=True, channels="GRAY")
else:
    st.image(cv2.cvtColor(edited, cv2.COLOR_BGR2RGB), use_container_width=True)

st.download_button(
    "Download Edited Image",
    data=cv2.imencode('.png', edited)[1].tobytes(),
    file_name="edited_image.png",
    mime="image/png"
)




