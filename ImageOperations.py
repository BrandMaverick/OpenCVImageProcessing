import cv2
import numpy as np
from scipy import ndimage 

def main():
    img1_src = "C:\\Users\\Sanjay Garg\\OneDrive\\Desktop\\Code\\ImageProcessing\\Images\\input-image-for-demo-throughout-1024x682.jpg"
    img2_src = "C:\\Users\\Sanjay Garg\\OneDrive\\Desktop\\Code\\ImageProcessing\\Images\\gray.png"
    img_output_src = "C:\\Users\\Sanjay Garg\\OneDrive\\Desktop\\Code\\ImageProcessing\\Images\\output "

    '''
    operation = "Image Arithmetic"
    output_img = add_images(img1_src, img2_src)
    write_image(img_output_src, operation, output_img)
    print_image("Added Image", output_img)

    operation = "Point Operations"
    output_img = threshold_image(img1_src)
    write_image(img_output_src, operation, output_img)
    print_image("Threshold Image", output_img)

    operation = "Geometric Operations"
    output_img = rotated_image(img1_src)
    write_image(img_output_src, operation, output_img)
    print_image("Rotated Image", operation)
    '''
    operation = "Morphology"
    output_img = dilate_image(img1_src)
    write_image(img_output_src, operation, output_img)
    print_image("Dilated Image", output_img)

    operation = "Digital Filters"
    output_img = mean_filter_image(img1_src)
    write_image(img_output_src, operation, output_img)
    print_image("Mean Filter Image", output_img)

    operation = "Feature Detectors"
    img_op_src = img_output_src + operation + ".jpg"
    roberts_cross_edge_detection(img1_src, img_op_src)
    print_image("Robert Cross Edge Image", load_image(img_op_src))


def write_image(img_output_src, operation, output_img):
    cv2.imwrite(img_output_src + operation + ".jpg", output_img)

def roberts_cross_edge_detection(img_src, img_output_src):
    img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE).astype('float64')
    img /= 255.0
    
    roberts_cross_v = np.array([[1, 0], [0, -1]])
    roberts_cross_h = np.array([[0, 1], [-1, 0]])
    
    vertical = ndimage.convolve(img, roberts_cross_v)
    horizontal = ndimage.convolve(img, roberts_cross_h)
    
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    edged_img *= 255
    
    cv2.imwrite(img_output_src, edged_img.astype(np.uint8))

def mean_filter_image(img_src):
    img = load_image(img_src)
    kernel_size = (5,5)
    kernel = np.ones(kernel_size,np.float32) / (kernel_size[0] + kernel_size[1])
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img

def dilate_image(img_src):
    img = load_image(img_src)
#    img = load_image_param(img_src, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5), np.uint8)
    dilated_img = cv2.erode(img, kernel, iterations=1)
    return dilated_img


def rotated_image(img_src):
    img = load_image(img_src)
    output_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return output_img


def threshold_image(img_src):
    img = load_image(img_src)
    th, output_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    return output_img

def add_images(img1_src, img2_src):
    # Ensure both images have the same dimensions
    img1 = load_image(img1_src)
    img2 = load_image(img2_src)

    if img1.shape != img2.shape:
        # Resize img2 to match img1's dimensions
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        added_image = cv2.add(img1, img2_resized)
    else:
        added_image = cv2.add(img1, img2)

    return added_image

def load_image(img_src):
    img = cv2.imread(img_src)
    # Ensure  image is loaded correctly
    if img is None:
        raise IOError("Error: One or both images could not be loaded. Please check the file paths.")
    return img

def load_image_param(img_src, param):
    img = cv2.imread(img_src, param)
    # Ensure  image is loaded correctly
    if img is None:
        raise IOError("Error: One or both images could not be loaded. Please check the file paths.")
    return img

def print_image(img_type, img):
    cv2.imshow(img_type, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




