import cv2
import numpy as np
import streamlit as st
from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt
from io import BytesIO

def threshold_region(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresholds = threshold_multiotsu(gray_image, classes=4)
    regions = np.digitize(gray_image, bins=thresholds)
    segm1 = (regions == 0).astype(np.uint8) * 255
    segm2 = (regions == 1).astype(np.uint8) * 255
    segm3 = (regions == 2).astype(np.uint8) * 255
    segm4 = (regions == 3).astype(np.uint8) * 255
    return segm1, segm2, segm3, segm4, regions

def display_image(image, caption):
    st.image(image, caption=caption, use_column_width=True)

def download_image(image, file_name):
    # Save the image to a BytesIO object
    image_buffer = BytesIO()
    plt.imsave(image_buffer, image, format="png")
    
    # Create a download link for the image
    st.download_button(
        label=f"Download {file_name}",
        data=image_buffer.getvalue(),
        file_name=file_name,
        mime="image/png"
    )

def main():
    st.title("Multi-Otsu Region Detection App")
    st.write("This app performs multi-Otsu thresholding on an uploaded image and displays the detected regions and segmented images.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        display_image(image, 'Uploaded Image')
        
        if st.button('Perform Multi-Otsu Thresholding'):
            segm1, segm2, segm3, segm4, regions = threshold_region(image)

            # Display the segmented images
            display_image(segm1, 'Segmented Region 1')
            download_image(segm1, "segmented_region_1.png")

            display_image(segm2, 'Segmented Region 2')
            download_image(segm2, "segmented_region_2.png")

            display_image(segm3, 'Segmented Region 3')
            download_image(segm3, "segmented_region_3.png")

            display_image(segm4, 'Segmented Region 4')
            download_image(segm4, "segmented_region_4.png")

            # Display the detected regions using matplotlib
            plt.figure(figsize=(10, 5))
            plt.imshow(regions, cmap="viridis")
            plt.title("Detected Regions")
            st.pyplot(plt)
            download_image(regions, "detected_regions.png")

if __name__ == '__main__':
    main()
