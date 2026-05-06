import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def manual_grayscale(image_bgr):
    b, g, r = image_bgr[:,:,0], image_bgr[:,:,1], image_bgr[:,:,2]
    gray = (0.114 * b) + (0.587 * g) + (0.299 * r)
    
    return gray.astype(np.uint8)

def manual_threshold(gray_image, threshold_value):
    binary_img = np.zeros_like(gray_image)
    binary_img[gray_image > threshold_value] = 255
    return binary_img

def manual_sobel_edges(gray_image, threshold=100):
    Kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]])
                   
    Ky = np.array([[1, 2, 1], 
                   [0, 0, 0], 
                   [-1, -2, -1]])
    
    
    Gx = convolve2d(gray_image, Kx, mode='same', boundary='symm')
    Gy = convolve2d(gray_image, Ky, mode='same', boundary='symm')
    
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    magnitude = (magnitude / magnitude.max()) * 255
    
    binary_edges = np.zeros_like(magnitude)
    binary_edges[magnitude > threshold] = 255
    
    return binary_edges.astype(np.uint8)

st.set_page_config(layout="wide", page_title="GNR 630: Circle Detection")

st.title("GNR 630: Circle Detection using Hough Transform")
st.markdown("Upload an image, and use the sliders to tune the parameters. Note: Grayscale conversion is handled via custom NumPy matrix operations to demonstrate fundamental pixel manipulation.")

st.sidebar.header("Tuning Parameters")

st.sidebar.subheader("Manual Edge Threshold")
edge_thresh = st.sidebar.slider("Sobel Edge Threshold", 0, 255, 100)

st.sidebar.subheader("Hough Transform Parameters")
param1 = st.sidebar.slider("Param 1 (Internal Edge Threshold)", 10, 300, 100, help="Upper threshold for the internal Canny edge detector.")
param2 = st.sidebar.slider("Param 2 (Center Threshold)", 10, 100, 50, help="Accumulator threshold. Lower = more false circles. Higher = stricter circles.")
min_dist = st.sidebar.slider("Minimum Distance Between Centers", 1, 100, 40)
min_radius = st.sidebar.slider("Minimum Radius", 1, 100, 10)
max_radius = st.sidebar.slider("Maximum Radius", 10, 200, 100)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    output_img = img.copy()
    
    gray = manual_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    edges = manual_sobel_edges(blurred, threshold=edge_thresh)
  
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=min_dist, 
        param1=param1, 
        param2=param2, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )
    
    num_circles = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        num_circles = len(circles[0])
        for pt in circles[0, :]:
            x, y, r = pt[0], pt[1], pt[2]
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)

    st.success(f"Successfully detected {num_circles} circles!")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
        
    with col2:
        st.header("Edge Map")
        st.image(edges, use_column_width=True, clamp=True)
        
    with col3:
        st.header("Detected Circles")
        st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)

else:
    st.info("Please upload an image to begin.")