from PIL import Image
import numpy as np

# Load the image
image_path = 'C:/Users/7501/Desktop/MCW_Intern/IMAGES/23.png'
image = Image.open(image_path)

# Convert image to RGB (if it's grayscale or other format)
image = image.convert("RGB")

# Resize the image to 32x32 if it's not already (CIFAR-10 images are 32x32)
image = image.resize((32, 32))

# Convert the image to a numpy array
image_array = np.array(image)

# Normalize the image array (values between 0 and 1)
image_array = image_array / 255.0  # Normalize to 0-1 range for floating point precision
# Convert the numpy array to a list of lists (for C++ input)
image_matrix = image_array.tolist()

# Store the matrix in the correct format in the text file
with open("image_matrix_frog.txt", "w") as f:
    for row in image_matrix:
        # Format each pixel's RGB values as space-separated floats
        row_str = ' '.join([f"{val[0]:.6f} {val[1]:.6f} {val[2]:.6f}" for val in row])
        f.write(f"{row_str}\n")

print("Image matrix has been written to 'image_matrix_frog.txt'.")
