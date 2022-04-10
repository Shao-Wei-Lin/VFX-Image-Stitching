import numpy as np  
import cv2 

def project_cylinder(img, focal_length=1100):
    h = img.shape[0]
    w = img.shape[1]
    K = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]]) 

    # ones = np.ones((h, w))
    # x_cord = ones*np.arange(w) - 0.5*w
    # y_cord = (ones.T*np.arange(h)).T -0.5*h 

    # theta = np.arctan(x_cord/focal_length)
    # height = y_cord/np.sqrt(x_cord**2 + focal_length**2)

    # new_x = focal_length*theta
    # new_y = focal_length*height

    # xn = 0.5 * dstShape[1] + xt * f
    # yn = 0.5 * dstShape[0] + yt * f

    # uv = np.dstack((new_x, new_y))
    # print('uv.shape =', uv.shape)

    # warped = cv2.remap(img, uv[:, :, 0].astype(np.float32), uv[:, :, 1].astype(np.float32), \
    #     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # return warped 

    y_i, x_i = np.indices((h, w))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h*w, 3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w*h,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]
    # make sure warp coords only within image bounds
    B[(B[:, 0] < 0) | (B[:, 0] >= w) | (B[:, 1] < 0) | (B[:, 1] >= h)] = -1
    B = B.reshape(h, w, -1)

    mask = cv2.inRange(B[:, :, 1], 0, h-1.0)&cv2.inRange(B[:, :, 0], 0, w-1.0)
    warped = cv2.remap(img, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img2_fg = cv2.bitwise_and(warped, warped, mask = mask)
    s = np.sum(img2_fg, axis = 0)
    c = 0
    while(np.sum(s[c])==0):
        c += 1
    return img2_fg[:, c:-c, :]
