import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------
def compute_output_size_1d(input_array, kernel_array):
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    conv_length = input_length - kernel_length + 1
    return conv_length
# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))

# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------
def convolve_1d(input_array, kernel_array):
    length = compute_output_size_1d(input_array, kernel_array)
    convolution = [None] * length
    for i in range(length):
        vec_input = input_array[list(range(i, i + length))]
        dot_product = np.dot(vec_input, kernel_array)
        convolution[i] = dot_product
    return convolution
# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------
def compute_output_size_2d(input_matrix, kernel_matrix):
    input_height = np.size(input_matrix, 0)
    input_width = np.size(input_matrix, 1)
    kernel_height = np.size(kernel_matrix, 0)
    kernel_width = np.size(kernel_matrix, 1)
    dimensions = (input_height - kernel_height + 1, input_width - kernel_width + 1)
    return dimensions
# -----------------------------------------------
# Example:
input_matrix = np.array([[1, 2, 3, 4, 5],
                       [5, 4, 3, 2, 1]])
kernel_matrix = np.array([[1, 0, -1],
                        [1, 0, -1]])
print(compute_output_size_2d(input_matrix, kernel_matrix))

# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    # Tip: same tips as above, but you might need a nested loop here in order to
    # define which parts of the input matrix need to be multiplied with the kernel matrix.
    out_height, out_width = compute_output_size_2d(input_matrix, kernel_matrix)
    kernel_height = np.size(kernel_matrix, 0)
    kernel_width = np.size(kernel_matrix, 1)

    convolution = np.zeros((out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            # Select the patch of the input that sits under the kernel
            patch = input_matrix[i:i + kernel_height, j:j + kernel_width]
            # Multiply element-wise and sum
            convolution[i, j] = np.sum(patch * kernel_matrix)
    return convolution
# -----------------------------------------------
# Example:
input_matrix = np.array([[1, 2, 3, 4, 5],
                         [5, 4, 3, 2, 1],
                         [5, 8, 1, 4, 6]])
kernel_matrix = np.array([[1, 0, -1],
                          [1, 0, -1]])
convolute_2d(input_matrix, kernel_matrix)