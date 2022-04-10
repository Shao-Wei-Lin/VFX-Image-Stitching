import numpy as np 
import cv2, os, sys, random  
import concat, cylinder
from descriptor import * 
import argparse 

e_sph = 70 #edge_of_sphere

def process_command(): # settin parsers
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--images', '-i', help = 'input image folder path.')
    parsers.add_argument('--output', '-o', help = 'output image path. (default = full_img.png)', default = 'full_img.png')
    parsers.add_argument('--reshape_ratio', '-r', help = 'set reshape ratio of image. (default = 0.25)', default = 0.25)
    parsers.add_argument('--focal_length', '-f', help = 'set the focal length. (default = 1100)', default = 1100)
    parsers.add_argument('--crop_pixels', '-c', help = 'crop above and below, avoid sphere part. (default = 70)', default = 70)

    parsers.add_argument('--show_points', '-d', help = 'show detected points on each image and output.', action = 'store_true')
    parsers.add_argument('--show_pairs', '-p', help = 'show pair points on each two images and output.', action = 'store_true')

    return parsers.parse_args()

def load_img(dir_path, focal_length, reshape_ratio):
    fname_list = os.listdir(dir_path)
    fname_list = sorted(fname_list)
    imgs = []
    for fname in fname_list:
        img = cv2.imread(os.path.join(dir_path, fname))[:, :, ::-1]
        img = cv2.resize(img, (int(img.shape[1] * reshape_ratio), int(img.shape[0] * reshape_ratio)), cv2.INTER_AREA)
        
        img = cylinder.project_cylinder(img, focal_length)
        if e_sph < 20:
            img = img[e_sph:-e_sph, e_sph:-e_sph]
        else:
            img = img[e_sph:-e_sph]
        imgs.append(img)
        print(f'Loading and warping {fname}...  ', end = '\r')
    print('Finish loading images                ')
    return imgs 

def detector(img, k = 0.04, point_num = 250): # by using Harris corner detection
    img = rgb2gray(img)

    x_grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
    y_grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)
    I_x2 = x_grad ** 2
    I_y2 = y_grad ** 2
    I_xy = x_grad * y_grad 
    I_x2 = cv2.GaussianBlur(I_x2, (3,3), cv2.BORDER_DEFAULT)
    I_y2 = cv2.GaussianBlur(I_y2, (3,3), cv2.BORDER_DEFAULT)
    I_xy = cv2.GaussianBlur(I_xy, (3,3), cv2.BORDER_DEFAULT)
    
    R = (I_x2*I_y2 - I_xy ** 2) - k * (I_x2 + I_y2) ** 2
    sorted_idx = np.unravel_index(np.argsort(-R, axis=None), R.shape) # sort from big to small
    sorted_idx = np.array(sorted_idx).T[:3000]
    #print(f"Rmax = {R[sorted_idx[0, 0], sorted_idx[0, 1]]}")
    #print(f"Rmin = {R[sorted_idx[-1, 0], sorted_idx[-1, 1]]}")
    
    points = sorted_idx[0].reshape(1, 2)
    radius_limit = 10000
    while radius_limit:
        for i in range(1, sorted_idx.shape[0]):
            place = sorted_idx[i].reshape(1, 2)
            min_dist = np.min(np.sum((points - place) ** 2, axis = 1))
            if(min_dist > radius_limit):
                points = np.vstack((points, place))
        if(points.shape[0] >= point_num):
            #print(radius_limit, points.shape[0])
            return points 
        radius_limit /= 2

def vis_corner_points(img, points, filename = 'default_vis.png'):
    img_opt = np.copy(img)
    for i in points:
        cv2.circle(img_opt, (i[1], i[0]), 5, (0, 255, 255), 2)
    cv2.imwrite(filename, img_opt[:, :, ::-1])

if __name__ == '__main__':
    args = process_command()
    dir_path = args.images  
    e_sph = int(args.crop_pixels)
    opt_filename = args.output
    focal_length = int(args.focal_length)

    crop = True # crop after loading
    imgs = load_img(dir_path, focal_length = focal_length, reshape_ratio = args.reshape_ratio) # imgs is a list
    vector_list = []
    
    move_list = []
    for i, img in enumerate(imgs):
        points = detector(img)
        
        vectored_points = descriptor(points, img)
        vectored_points = np.hstack((points, vectored_points.reshape(-1, 64)))   

        vector_list.append(vectored_points)
        if args.show_points:
            vis_corner_points(img, points, filename = f'img_corner_vis{str(i).zfill(2)}.png')
        
        if i != 0:
            pair_points, dist_pair = concat.match_points(vector_list[i-1], vector_list[i]) 
            move, main_pair = concat.move_amount(pair_points, dist_pair)
            move_list.append(move)
            if args.show_pairs:
                concat.vis_pair(imgs[i-1], imgs[i], pair_points, main_pair = main_pair, filename = f'pair_{i-1}_{i}.png')
        print(f'Finishing extract feature of images {i+1}/{len(imgs)}', end = '\r')
    final_img = concat.stitch(imgs, move_list)
    cv2.imwrite(opt_filename, final_img[:, :, ::-1])   
    print(f'Finished                                                    ')


