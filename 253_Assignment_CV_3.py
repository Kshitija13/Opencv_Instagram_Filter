import streamlit as st
from PIL import Image
import cv2
import numpy as np

#Bluring filter
def Average_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / 25
    dst = cv2.filter2D(image, -1, kernel)
    return dst

#cartoon effect
#Colour Quantization
def ColourQuantization(image, K=9):
    Z = image.reshape((-1, 3)) 
    Z = np.float32(Z) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    compactness, label, center = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2

#to get countours
def Countours(image):
    contoured_image = image
    gray = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(gray, 200, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    cv2.drawContours(contoured_image, contours, contourIdx=-1, color=6, thickness=1)
    return contoured_image

#greyscale filter
def greyscale(img):
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return greyscale

# brightness adjustment
def bright(img, beta_value ):
    img_bright = cv2.convertScaleAbs(img, beta=beta_value)
    return img_bright

#sharp effect
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen

#sepia effect
def sepia(img):
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia


#grey pencil sketch effect
def pencil_sketch_grey(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return  sk_gray

#colour pencil sketch effect
def pencil_sketch_col(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return  sk_color


#HDR effect
def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return  hdr

# invert filter
def invert(img):
    inv = cv2.bitwise_not(img)
    return inv

#defining a function
from scipy.interpolate import UnivariateSpline
def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

#summer effect
def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum

#winter effect
def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win

def Insta_Filter():
    st.title("CV Assignment 1")
    st.subheader("Create your own Instagram Filter.")
    st.text("Name: Kshitija Lade")
    st.text("Roll No.: 253")
    st.text("PRN: 0120190090")
    st.text("Batch: CV2")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    st.image(original_image, caption="★ Original Image ★")


    filter = st.radio(
        "**********Choose your Favourite Filter**********",
        ["Cartoon Effect", "Blurring Filter", "GreySacle", "Sharpen", "Sepia Filter", "Pencil_Sketch_Grey", "Pencil_Sketch_Col", "HDR", "Invert", "Summer",
         "Winter"],
        key="filter"
    )

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',unsafe_allow_html=True)

    brightness_amount = st.slider("Brightness", min_value=-50, max_value=50, value=0)
    processed_image = bright(original_image, brightness_amount)

    if filter == "GreySacle":
        processed_image = greyscale(processed_image)

    elif filter == "Cartoon Effect":
        coloured = ColourQuantization(processed_image)
        contoured = Countours(coloured)
        processed_image = contoured

    elif filter == "Blurring Filter":
        kernel_size = st.slider("Kernel", min_value=3, max_value=15)
        processed_image = Average_filter(processed_image, kernel_size)

    elif filter == "Sharpen":
        processed_image = sharpen(processed_image)

    elif filter == "Sepia Filter":
        processed_image = sepia(processed_image)

    elif filter == "Pencil_Sketch_Grey":
        processed_image = pencil_sketch_grey(processed_image)

    elif filter == "Pencil_Sketch_Col":
        processed_image = pencil_sketch_col(processed_image)

    elif filter == "HDR":
        processed_image = HDR(processed_image)

    elif filter == "Invert":
        processed_image == invert(processed_image)

    elif filter == "Summer":
        processed_image = Summer(processed_image)

    elif filter == "Winter":
        processed_image = Winter(processed_image)

    else:
        st.text("Sorry Filter is not Available")

    label = "***************Result of %s Filter***************" % filter
    st.image(processed_image, caption=label)


def main_loop():
    Insta_Filter()

if __name__ == "__main__":
    main_loop()






