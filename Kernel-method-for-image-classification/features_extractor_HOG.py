import numpy as np

def grad_image(image):
    ### compute the discrete derivative gradient of an image
    H = 32
    W = 32
    grad_x = np.zeros((H,W,3))
    grad_y = np.zeros((H,W,3))
    # Contour
    # Initialize boundaries to zero
    grad_x[:,0,:] = 0
    grad_x[:,-1,:] = 0
    grad_y[0,:,:] = 0
    grad_y[-1,:,:] = 0

    # Compute discrete derivative of the true image
    grad_x[:,1:-1,:] = image[:,2:,:] - image[:,:-2,:]
    grad_y[1:-1,:,:] = image[2:,:,:] - image[:-2,:,:]

    return grad_x, grad_y


def update_cell_hog(magnitude, orientation, orientation_start, orientation_end):
    ### compute magnitude of a cell
    tot = 0
    for i in range(8):
        for j in range(8):
            if (orientation[i, j] >= orientation_start) or (orientation[i, j] < orientation_end):
                continue
            tot += magnitude[i, j]
    return tot / (8 * 8)

def orient_hist(magnitude, orient_hist, c_col, c_row, n_cells_row, n_cells_col, orientations):
    ### compute orientation histogram for a cell
    hist = np.zeros((n_cells_row, n_cells_col, orientations))
    r_0 = c_row / 2
    c_0 = c_col / 2
    cc = c_row * n_cells_row
    cr = c_col * n_cells_col
    range_rows_stop = (c_row + 1) / 2
    range_rows_start = -(c_row / 2)
    range_columns_stop = (c_col + 1) / 2
    range_columns_start = -(c_col / 2)

    # Iterate over orientations
    for i in range(orientations):
        orientation_start = 180 * (i + 1) / orientations
        orientation_end = 180 * i / orientations
        c = c_0
        r = r_0
        r_i = 0
        c_i = 0

        while r < cc:
            c_i = 0
            c = c_0

            while c < cr:
                block_magnitude = magnitude[int(r+range_rows_start):int(r+range_rows_stop), int(c+range_columns_start):int(c+range_columns_stop)]
                block_orientation = orient_hist[int(r+range_rows_start):int(r+range_rows_stop), int(c+range_columns_start):int(c+range_columns_stop)]
                hist[r_i, c_i, i] = update_cell_hog(block_magnitude, block_orientation, orientation_start, orientation_end)

                c_i += 1
                c += c_col

            r_i += 1
            r += c_row
    return hist

def hog(image, div):
    ### return the hog features of the image as a vector

    # Supposed image is size 32x32
    H = 32
    W = 32
    grad_x, grad_y = grad_image(image)
    magn = np.zeros(grad_x.shape)
    for pix in range(3):
        magn[:,:,pix] = np.sqrt(grad_x[:,:,pix]**2 + grad_y[:,:,pix]**2)

    # We take those with highest magnitude
    idcs_max = magn.argmax(axis=2)
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij', sparse=True)

    c_row, c_col = (div, div) # pixels_per_cell

    n_cells_row = int(H // c_row)
    n_cells_col = int(W // c_col)

    # Compute orientation histogram
    orientations = 9
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations))
    angle = np.arctan2(grad_y , grad_x)
    orientation_histogram = np.rad2deg(angle) % 180
    # Take those with highest magnitude
    new_magn = magn[rr, cc, idcs_max]
    new_orient  = orientation_histogram[rr, cc, idcs_max]

    hist = orient_hist(new_magn, new_orient, c_col, c_row, n_cells_row, n_cells_col, orientations)

    return hist.ravel()

def get_hog_feature(data, div =8):
    ### Extract HOG features from a dataset
    size_sample = data.shape[0]
    features = np.zeros((size_sample, int((32//div)**2 * 9)))
    for i in range(size_sample):
        features[i, :] = hog(data[i], div)      
    return features
