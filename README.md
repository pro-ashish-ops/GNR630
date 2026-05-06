# Circle Detection via Hough Transform 
**GNR 630: Image Processing & Analysis**

An interactive computer vision pipeline designed to detect circular geometric primitives (such as coins and washers) in high-resolution imagery. This project features a custom-built Streamlit dashboard for real-time parameter tuning and visual feedback.

## ⚙️ Algorithm Pipeline
To demonstrate an understanding of fundamental image matrix manipulations, the preprocessing stages bypass standard library calls in favor of pure mathematical implementations:
1. **Manual Grayscale Conversion:** Vectorized NumPy implementation using the standard human luminosity formula.
2. **Manual Gaussian Blur:** Dynamic 2D kernel generation and convolution to act as a low-pass filter, suppressing high-frequency surface noise.
3. **Manual Sobel Operator:** Explicit $3 \times 3$ kernel matrix convolution to extract spatial gradients and directions, followed by binary thresholding.
4. **Hough Accumulator:** Center and radius detection using the optimized `cv2.HOUGH_GRADIENT` method.

## 📚 Academic References & Theoretical Foundation
The circle detection in this project relies on the **2-Stage Gradient Method**, which drastically reduces the computational load of the standard Hough Transform by bypassing the 3D parameter space. Instead, it uses the spatial gradient direction of the edge pixels to cast votes into a 2D accumulator to find the center, followed by a 1D histogram to estimate the radius.

The mathematical foundation and efficiency proofs for this specific approach are detailed in the following literature:

1. **The 2-Stage Method (OpenCV Implementation Basis):**
   > Yuen, H. K., Princen, J., Illingworth, J., & Kittler, J. (1990). *Comparative study of Hough Transform methods for circle finding*. Image and Vision Computing, 8(1), 71-77. 
   > https://drive.google.com/file/d/1J05tZB9R5e5V7LJ5i-7GOO0RGQ031W-_/view?usp=sharing

2. **The Foundational Gradient Constraint:**
   > Kimme, C., Ballard, D., & Sklansky, J. (1975). *Finding circles by an array of accumulators*. Communications of the ACM, 18(2), 120-122.
   > https://drive.google.com/file/d/1iHNVs8Z0tB93iz80TF_sxcZe2UmasmBy/view?usp=sharing

## 🚀 How to Run the Application
1. Clone this repository:
   ```bash
   git clone [https://github.com/pro-ashish-ops/GNR630.git]
