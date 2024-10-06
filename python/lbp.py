import numpy as np
from skimage import io
from scipy import ndimage

def LBP(image, radius=1, neighbors=8, mapping=None, mode='h'):
    def roundn(x, n):
        if n < 0:
            p = 10 ** -n
            return np.round(p * x) / p
        elif n > 0:
            p = 10 ** n
            return p * np.round(x / p)
        else:
            return np.round(x)

    if isinstance(image, str):
        image = io.imread(image, as_gray=True)
    
    if len(image.shape) > 2:
        raise ValueError("Input image should be grayscale")
    
    ysize, xsize = image.shape
    
    if neighbors == 8:
        spoints = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    else:
        spoints = np.zeros((neighbors, 2))
        a = 2 * np.pi / neighbors
        for i in range(neighbors):
            spoints[i, 0] = -radius * np.sin(i * a)
            spoints[i, 1] = radius * np.cos(i * a)
    
    miny = min(spoints[:, 0])
    maxy = max(spoints[:, 0])
    minx = min(spoints[:, 1])
    maxx = max(spoints[:, 1])
    
    bsizey = int(np.ceil(max(maxy, 0))) - int(np.floor(min(miny, 0))) + 1
    bsizex = int(np.ceil(max(maxx, 0))) - int(np.floor(min(minx, 0))) + 1
    
    origy = 1 - int(np.floor(min(miny, 0)))
    origx = 1 - int(np.floor(min(minx, 0)))
    
    if xsize < bsizex or ysize < bsizey:
        raise ValueError("Too small input image. Should be at least (2*radius+1) x (2*radius+1)")
    
    dx = xsize - bsizex + 1
    dy = ysize - bsizey + 1
    
    # Pad the image to handle border cases
    padded_image = np.pad(image, ((radius, radius), (radius, radius)), mode='edge')
    
    # Adjust C extraction to use padded image
    C = padded_image[radius+origy-1:radius+origy+dy-1, radius+origx-1:radius+origx+dx-1]
    
    bins = 2**neighbors
    
    result = np.zeros((dy, dx), dtype=np.uint64)
    
    for i in range(neighbors):
        y, x = spoints[i]
        y = y + radius + origy
        x = x + radius + origx
        fy = int(np.floor(y))
        cy = int(np.ceil(y))
        ry = int(np.round(y))
        fx = int(np.floor(x))
        cx = int(np.ceil(x))
        rx = int(np.round(x))
        
        if abs(x - rx) < 1e-6 and abs(y - ry) < 1e-6:
            N = padded_image[ry-1:ry+dy-1, rx-1:rx+dx-1]
        else:
            ty = y - fy
            tx = x - fx
            
            w1 = roundn((1 - tx) * (1 - ty), -6)
            w2 = roundn(tx * (1 - ty), -6)
            w3 = roundn((1 - tx) * ty, -6)
            w4 = roundn(1 - w1 - w2 - w3, -6)
            
            N = (w1 * padded_image[fy-1:fy+dy-1, fx-1:fx+dx-1] +
                 w2 * padded_image[fy-1:fy+dy-1, cx-1:cx+dx-1] +
                 w3 * padded_image[cy-1:cy+dy-1, fx-1:fx+dx-1] +
                 w4 * padded_image[cy-1:cy+dy-1, cx-1:cx+dx-1])
            
            N = roundn(N, -4)
        
        D = (N >= C).astype(np.uint64)
        v = 2**i
        result += v * D
    
    if mapping is not None:
        bins = mapping['num']
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = mapping['table'][result[i, j]]
    
    if mode in ['h', 'hist', 'nh']:
        result = np.histogram(result.flatten(), bins=range(bins + 1))[0]
        if mode == 'nh':
            result = result / np.sum(result)
    else:
        if bins - 1 <= 255:
            result = result.astype(np.uint8)
        elif bins - 1 <= 65535:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint32)
    
    return result

# Example usage
if __name__ == "__main__":
    image = io.imread('path_to_your_image.png', as_gray=True)
    lbp_result = LBP(image, radius=1, neighbors=8, mode='h')
    print(lbp_result)