import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_filter_from_file(file_path):
    # Read the filter from a file
    filter = np.loadtxt(file_path)
    return filter

def create_gaussian_filter(size, sigma):
    # Create a Gaussian filter with the given size and sigma
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Choose either to read from a file or create a Gaussian filter
filter_option = "gaussian"  # "file" or "gaussian"
filter_file_path = "your_filter.txt"  # Specify your filter file path if using a file
filter_size = 9  # M
sigma = filter_size / 4.0

# Load the input image (N x N image)
input_image = cv2.imread("pic1grey300.jpg", cv2.IMREAD_GRAYSCALE)
input_image2 = cv2.imread("pic2grey300.jpg", cv2.IMREAD_GRAYSCALE)


def apply_convolution(input_image, filter):
    M = filter.shape[0]
    N = input_image.shape[0]
    output_image = np.zeros((N, N), dtype=np.float32)

    for i in range((M - 1) // 2, N - 1-(M - 1) // 2):
        for j in range((M - 1) // 2, N - 1-(M - 1) // 2):
            # Extract the region of interest from the input image
            roi = input_image[i - (M - 1) // 2:i + (M - 1) // 2 + 1, j - (M - 1) // 2:j + (M - 1) // 2 + 1]
            # Apply convolution
            output_image[i, j] = np.sum(roi * filter)

    return output_image

if filter_option == "file":
    custom_filter = read_filter_from_file(filter_file_path)
else:
    custom_filter = create_gaussian_filter(filter_size, sigma)
#***************************************Part 1**************************************************************
# Apply convolution to the image
output_image = apply_convolution(input_image, custom_filter)
output_image2 = apply_convolution(input_image2, custom_filter)

# Normalize the output image to the 0-255 range
output_image_normalized = output_image / np.sum(output_image)
output_image_normalized2 = output_image2 / np.sum(output_image2)

# Display input and output images
plt.subplot(121), plt.imshow(input_image, cmap='gray'), plt.title('Input Image')
plt.subplot(122), plt.imshow(output_image_normalized, cmap='gray'), plt.title('Output Image')
plt.show()
plt.subplot(121), plt.imshow(input_image2, cmap='gray'), plt.title('Input Image 2')
plt.subplot(122), plt.imshow(output_image_normalized2, cmap='gray'), plt.title('Output Image 2')
plt.show()
cv2.imwrite('task1_output1.jpg', output_image)
cv2.imwrite('task1_output2.jpg', output_image2)
#***************************************Part 2**************************************************************
input_image = cv2.imread("pic1grey300.jpg", cv2.IMREAD_GRAYSCALE)
input_image2 = cv2.imread("pic2grey300.jpg", cv2.IMREAD_GRAYSCALE)
def compute_gaussian_filter(sigma, M):
    g = np.zeros(M)
    summ = 0
    for k in range(M):
        g[k] = np.exp(-((k - (M - 1) / 2) ** 2) / (2 * sigma ** 2))
        summ += g[k]
    
    # Normalize the filter coefficients
    g = g / summ
    return g

def apply_gaussian_filter(image, filter):
    N = image.shape[0]
    M = len(filter)
    output = np.zeros((N, N), dtype=np.float32)
    
    # Filter each row
    for i in range(N):
        for j in range((M - 1) // 2, N - 1-(M - 1) // 2):
            summ = 0
            for k in range(M):
                summ += filter[k] * image[i, j - (k - (M - 1) // 2)]
            output[i, j] = summ

    # Filter each column
    for j in range(N):
        for i in range((M - 1) // 2, N - 1-(M - 1) // 2):
            summ = 0
            for k in range(M):
                summ += filter[k] * output[i - (k - (M - 1) // 2), j]
            output[i, j] = summ

    return output

# Define sigma and filter size
sigma_values = [1.0, 2.0, 3.0]
filter_size = 9

# Apply Gaussian filter for different sigma values
smoothed_images = []
smoothed_images2 = []
for sigma in sigma_values:
    gaussian_filter = compute_gaussian_filter(sigma, filter_size)
    smoothed_image = apply_gaussian_filter(input_image, gaussian_filter)
    smoothed_images.append(smoothed_image)
    gaussian_filter2 = compute_gaussian_filter(sigma, filter_size)
    smoothed_image2 = apply_gaussian_filter(input_image2, gaussian_filter2)
    smoothed_images2.append(smoothed_image2)
    

def compute_gradient_magnitude(image):
    gradient_x = np.zeros(image.shape, dtype=np.float32)
    gradient_y = np.zeros(image.shape, dtype=np.float32)
    
    # Compute the gradient in the x-direction
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            gradient_x[i, j] = image[i, j + 1] - image[i, j - 1]

    # Compute the gradient in the y-direction
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            gradient_y[i, j] = image[i + 1, j] - image[i - 1, j]

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

def threshold_edges(gradient_magnitude, threshold):
    edge_image = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    edge_image[gradient_magnitude >= threshold] = 255
    return edge_image

# Define threshold values
threshold_values = [20, 30, 40, 50]

# Perform edge detection for different sigma and threshold values
edge_images = []
edge_images2 = []

for smoothed_image in smoothed_images:
    for threshold in threshold_values:
        gradient_magnitude = compute_gradient_magnitude(smoothed_image)
        edge_image = threshold_edges(gradient_magnitude, threshold)
        edge_images.append(edge_image)
        
for smoothed_image2 in smoothed_images2:
    for threshold in threshold_values:
        gradient_magnitude2 = compute_gradient_magnitude(smoothed_image2)
        edge_image2 = threshold_edges(gradient_magnitude2, threshold)
        edge_images2.append(edge_image2)

# Display the output images

task2output = ['task2_output1.jpg','task2_output2.jpg','task2_output3.jpg','task2_output4.jpg','task2_output5.jpg','task2_output6.jpg','task2_output7.jpg','task2_output8.jpg','task2_output9.jpg','task2_output10.jpg','task2_output11.jpg','task2_output12.jpg']
task2output2 = ['task2_im2_output1.jpg','task2_im2_output2.jpg','task2_im2_output3.jpg','task2_im2_output4.jpg','task2_im2_output5.jpg','task2_im2_output6.jpg','task2_im2_output7.jpg','task2_im2_output8.jpg','task2_im2_output9.jpg','task2_im2_output10.jpg','task2_im2_output11.jpg','task2_im2_output12.jpg']
for i, edge_image in enumerate(edge_images):
    plt.subplot(len(sigma_values), len(threshold_values), i + 1)
    plt.imshow(edge_image, cmap='gray')
    plt.title(f"Sigma={sigma_values[i // len(threshold_values)]}, Threshold={threshold_values[i % len(threshold_values)]}")
    #cv2.imwrite(task2output[i], edge_image)
plt.show()
for i, edge_image2 in enumerate(edge_images2):
    plt.subplot(len(sigma_values), len(threshold_values), i + 1)
    plt.imshow(edge_image2, cmap='gray')
    plt.title(f"Sigma={sigma_values[i // len(threshold_values)]}, Threshold={threshold_values[i % len(threshold_values)]}")
    #cv2.imwrite(task2output2[i], edge_image2)
plt.show()


#***************************************Part 3A**************************************************************
# Load the input image
input_image = cv2.imread("pic1grey300.jpg", cv2.IMREAD_GRAYSCALE)
input_image2 = cv2.imread("pic2grey300.jpg", cv2.IMREAD_GRAYSCALE)
# Define the window size for computing corner metrics
window_size = 11
sigma = 5.5
#corner_threshold = 800

# step 1: Apply Gaussian smoothing
sigma_smoothing = 2.0
gaussian_filter = compute_gaussian_filter(sigma_smoothing, 9)
image = apply_gaussian_filter(input_image, gaussian_filter)
image2 = apply_gaussian_filter(input_image2, gaussian_filter)
                          
# step 2: Compute gradient vectors (Ix, Iy)
Ix = np.zeros(image.shape, dtype=np.float32)
Iy = np.zeros(image.shape, dtype=np.float32)
    
# Compute the gradient in the x-direction
for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):
        Ix[i, j] = image[i, j + 1] - image[i, j - 1]

# Compute the gradient in the y-direction
for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):
        Iy[i, j] = image[i + 1, j] - image[i - 1, j]
Ix = Ix/10
Iy=Iy/10

            
Ix2 = np.zeros(image2.shape, dtype=np.float32)
Iy2 = np.zeros(image2.shape, dtype=np.float32)
    
# Compute the gradient in the x-direction
for i in range(1, image2.shape[0] - 1):
    for j in range(1, image2.shape[1] - 1):
        Ix2[i, j] = image2[i, j + 1] - image2[i, j - 1]

# Compute the gradient in the y-direction
for i in range(1, image2.shape[0] - 1):
    for j in range(1, image2.shape[1] - 1):
        Iy2[i, j] = image2[i + 1, j] - image2[i - 1, j]
Ix2 = Ix2/10
Iy2=Iy2/10

# step 3: Compute A, B, and C
A = Ix * Ix
B = Iy * Iy
C = Ix * Iy

A2 = Ix2 * Ix2
B2 = Iy2 * Iy2
C2 = Ix2 * Iy2
                          
# step 4: Apply Gaussian smoothing to A, B, and C
sigma_corner = 5.5
gaussian_filter_corner = compute_gaussian_filter(sigma_corner, 11)
A_smoothed = apply_gaussian_filter(A, gaussian_filter_corner)
B_smoothed = apply_gaussian_filter(B, gaussian_filter_corner)
C_smoothed = apply_gaussian_filter(C, gaussian_filter_corner)

A_smoothed2 = apply_gaussian_filter(A2, gaussian_filter_corner)
B_smoothed2 = apply_gaussian_filter(B2, gaussian_filter_corner)
C_smoothed2 = apply_gaussian_filter(C2, gaussian_filter_corner)
                          
# step 6: Compute the Harris corner response
R = (A_smoothed * B_smoothed - C_smoothed * C_smoothed) - 0.04 * (A_smoothed + B_smoothed) ** 2
max_response = np.max(R)
corner_threshold = 0.01 * max_response

R2 = (A_smoothed2 * B_smoothed2 - C_smoothed2 * C_smoothed2) - 0.04 * (A_smoothed2 + B_smoothed2) ** 2
max_response2 = np.max(R2)
corner_threshold2 = 0.01 * max_response2
                          
# step 7: Threshold the corner response
corner_image = np.zeros_like(R, dtype=np.uint8)
corner_image[R > corner_threshold] = 255

corner_image2 = np.zeros_like(R2, dtype=np.uint8)
corner_image2[R2 > corner_threshold2] = 255
                          
# step 8: Non-maxima suppression
corners = []
for i in range(1, R.shape[0] - 1):
    for j in range(1, R.shape[1] - 1):
        if R[i, j] > corner_threshold:
            local_max = np.max(R[i-1:i+2, j-1:j+2])
            if R[i, j] == local_max:
                corners.append((j, i))  # (x, y) format

corners2 = []
for i in range(1, R2.shape[0] - 1):
    for j in range(1, R2.shape[1] - 1):
        if R2[i, j] > corner_threshold2:
            local_max2 = np.max(R2[i-1:i+2, j-1:j+2])
            if R2[i, j] == local_max2:
                corners2.append((j, i))  # (x, y) format
                
# Display the original image with detected corner points

output_image2 = cv2.cvtColor(input_image2, cv2.COLOR_GRAY2BGR)
for corner in corners2:
    x, y = corner
    cv2.circle(output_image2, (x, y), 3, (0, 0, 255), -1)  # Draw corners as red circles
cv2.imwrite("task3_output2.jpg", output_image2)
cv2.imshow("Corner Detection2", output_image2)

output_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
for corner in corners:
    x, y = corner
    cv2.circle(output_image, (x, y), 3, (0, 0, 255), -1)
cv2.imwrite("task3_output1.jpg", output_image)
cv2.imshow("Corner Detection", output_image)

#***************************************Part 3B**************************************************************
gradient_scale = 0.1
num_bins = 8

normalized_histograms = []
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    gradient_magnitudes = np.sqrt(Ix[y-4:y+5, x-4:x+5]**2 + Iy[y-4:y+5, x-4:x+5]**2)
    gradient_directions = np.arctan2(Iy[y-4:y+5, x-4:x+5], Ix[y-4:y+5, x-4:x+5])
    
    histogram = np.zeros(num_bins, dtype=np.float32)
    
    for direction, magnitude in zip(gradient_directions.ravel(), gradient_magnitudes.ravel()):
        angle = (np.degrees(direction) + 360) % 360
        bin_index = int(angle / 45) % num_bins
        histogram[bin_index] += magnitude

    # Find the bin with the maximum value
    max_bin_index = np.argmax(histogram)
    
    # Normalize the histogram
    normalized_histogram = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        normalized_histogram[(4 + i) % num_bins] = histogram[(max_bin_index + i) % num_bins]
    
    normalized_histograms.append(normalized_histogram)
    
# Display the corners and their normalized histograms on the original image
for corner, normalized_histogram in zip(corners, normalized_histograms):
    x, y = corner.ravel()
    print("pixel at (i,j)=(",x,",",y,") has histogram ",normalized_histogram)
    print("")


