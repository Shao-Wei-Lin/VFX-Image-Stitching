import numpy as np  
import cv2 

def rgb2gray(img):
    return np.sum(img * np.array([0.2990, 0.5870, 0.1140]).reshape(1, 1, 3), axis = -1)

def extract_patch(img, point, size):
    img = np.pad(img, int(size/2), 'constant')
    patch_x = point[0]
    patch_y = point[1]
    return img[patch_x:patch_x+size, patch_y:patch_y+size]

def create_gaussian(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2) ))
    return g

def get_orientation(points, m, theta, size = 8):
    orientation = np.empty(points.shape[0])
    bin_gap = np.pi/18
    gaussian_filter = create_gaussian(size)
    for n, point in enumerate(points):
        bin = np.zeros(36)
        patched_m = extract_patch(m, point, size)
        patched_theta = extract_patch(theta, point, size)
        weighted_m = patched_m * gaussian_filter 
        for i in range(size):
            for j in range(size):
                bin[int(patched_theta[i][j]/bin_gap)%36] += weighted_m[i][j]
        orientation[n] = np.argmax(bin)
    return orientation

def descriptor(points, img, orientation=True):
    # get m(x,y) and theta(x,y)
    img = rgb2gray(img)

    x_grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
    y_grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)

    if orientation:
        m = np.sqrt( np.square(x_grad) + np.square(y_grad))
        theta = np.arctan(y_grad/(x_grad+1e-10))
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                if x_grad[i][j] < 0:
                    if y_grad[i][j] < 0:
                        theta[i][j] -= np.pi
                    else:
                        theta[i][j] += np.pi
                if theta[i][j] < 0:
                    theta[i][j] += 2*np.pi

        # get orientation of the feature points
        orientations = get_orientation(points, m, theta)

        # get resized image of feature points
        resized = np.empty([points.shape[0], 8, 8])
        degree = orientations*(-10) + 5
        for n, point in enumerate(points):
            center = (40, 40)
            patch = extract_patch(img, point, size=80)
            # print(f'degree {str(n).zfill(3)} =', degree[n])
            # cv2.imwrite(f'./patch_80/{str(n).zfill(3)}_patch.png', np.dstack((patch, patch, patch)))
            M = cv2.getRotationMatrix2D(center, degree[n], 1.0)
            rotated = cv2.warpAffine(patch, M, (80,80)) 
            # cv2.imwrite(f'./rotate_80/{str(n).zfill(3)}_rotate_patch.png', np.dstack((rotated, rotated, rotated)))
            patch = extract_patch(rotated, center, size = 40)
            # cv2.imwrite(f'./rotate_40/{str(n).zfill(3)}_rotate_patch.png', np.dstack((patch, patch, patch)))
            resized[n] = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_LINEAR)

    # print('resized array shape =', resized.shape)
    # print('resized image shape =', resized[0].shape);
    else:
        resized = np.empty([points.shape[0], 8, 8])
        for i, point in enumerate(points):
            patch = extract_patch(img, point, size=40)
            resized[n] = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_LINEAR)

    mean = np.mean(resized, axis=(1,2))
    std = np.std(resized, axis=(1,2))
    for n in range(mean.shape[0]):
        for i in range(8):
            for j in range(8):
                resized[n][i][j] = (resized[n][i][j] - mean[n])/std[n]

    return resized
