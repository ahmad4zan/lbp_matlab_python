import numpy as np
from scipy import ndimage

def LBP(image, radius=1, neighbors=8, mapping=None, mode='h'):
    """
    Compute Local Binary Pattern (LBP) of an image.

    Parameters:
    - image: Input image (2D numpy array)
    - radius: Radius of the circular pattern
    - neighbors: Number of sampling points
    - mapping: LBP mapping (not implemented in this version)
    - mode: 'h' for histogram, 'nh' for normalized histogram, otherwise returns LBP image

    Returns:
    - LBP histogram or LBP image depending on the mode
    """
    
    # Convert image to double
    image = image.astype(np.float64)
    
    # Determine the dimensions of the input image
    ysize, xsize = image.shape
    
    # Generate circular sampling points
    spoints = np.array([
        [-np.sin(2*np.pi*i/neighbors), np.cos(2*np.pi*i/neighbors)]
        for i in range(neighbors)
    ]) * radius
    
    # Determine the dimensions of the block
    miny, maxy = np.min(spoints[:, 0]), np.max(spoints[:, 0])
    minx, maxx = np.min(spoints[:, 1]), np.max(spoints[:, 1])
    
    # Block size
    bsizey = int(np.ceil(max(maxy, 0))) - int(np.floor(min(miny, 0))) + 1
    bsizex = int(np.ceil(max(maxx, 0))) - int(np.floor(min(minx, 0))) + 1
    
    # Coordinates of origin (0,0) in the block
    origy = 1 - int(np.floor(min(miny, 0)))
    origx = 1 - int(np.floor(min(minx, 0)))
    
    # Check minimum allowed size for the input image
    if xsize < bsizex or ysize < bsizey:
        raise ValueError("Too small input image. Should be at least (2*radius+1) x (2*radius+1)")
    
    # Calculate dx and dy
    dx = xsize - bsizex
    dy = ysize - bsizey
    
    # Fill the center pixel matrix C
    C = image[origy-1:origy+dy, origx-1:origx+dx]
    
    # Initialize the result matrix with zeros
    result = np.zeros((dy+1, dx+1), dtype=np.uint32)
    
    # Compute the LBP code image
    for i, (y, x) in enumerate(spoints):
        # Calculate coordinates of sample points
        y = y + origy
        x = x + origx
        
        # Interpolation of adjacent pixels
        fx, fy = np.floor(x).astype(int), np.floor(y).astype(int)
        cx, cy = np.ceil(x).astype(int), np.ceil(y).astype(int)
        ty, tx = y - fy, x - fx
        
        # Calculate interpolation weights
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty
        
        # Compute interpolated pixel values
        N = (w1 * image[fy:fy+dy+1, fx:fx+dx+1] +
             w2 * image[fy:fy+dy+1, cx:cx+dx+1] +
             w3 * image[cy:cy+dy+1, fx:fx+dx+1] +
             w4 * image[cy:cy+dy+1, cx:cx+dx+1])
        
        # Update the result matrix
        D = N >= C
        result += (1 << i) * D
    
    # Apply mapping if it is defined (not implemented in this version)
    if mapping is not None:
        raise NotImplementedError("Mapping is not implemented in this version")
    
    if mode in ['h', 'hist', 'nh']:
        # Return LBP histogram
        bins = 2**neighbors
        hist, _ = np.histogram(result.flatten(), bins=range(bins+1))
        if mode == 'nh':
            hist = hist / np.sum(hist)
        return hist
    else:
        # Return LBP image
        return result.astype(np.uint8)

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io
    
    # Load an image (replace with your image path)
    image = io.imread('path_to_your_image.png', as_gray=True)
    
    # Compute LBP
    lbp_hist = LBP(image, radius=1, neighbors=8, mode='h')
    
    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(lbp_hist)), lbp_hist)
    plt.title('LBP Histogram')
    plt.xlabel('LBP code')
    plt.ylabel('Frequency')
    plt.show()
    
    # Compute LBP image
    lbp_image = LBP(image, radius=1, neighbors=8, mode='i')
    
    # Display original and LBP images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(lbp_image, cmap='gray')
    ax2.set_title('LBP Image')
    plt.show()