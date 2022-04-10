import numpy as np
import sys, os, cv2, random

# descriptor of left and right image
def match_points(l, r):     
    place_l, vec_l = l[:, :2], l[:, 2:]
    place_r, vec_r = r[:, :2], r[:, 2:]
    
    pair_rec = []
    for l_p, l_v in zip(place_l, vec_l):
        min_dist = 1e10
        sec_min_dist = 1e10
        for r_p, r_v in zip(place_r, vec_r):
            if abs(l_p[0] - r_p[0]) < 50 and r_p[1] + 30 < l_p[1]:
                dist = (l_v - r_v) @ (l_v - r_v)  
                if dist < min_dist:
                    sec_min_dist = min_dist
                    min_dist = dist                
                    min_p = r_p 
                elif dist < sec_min_dist:
                    sec_min_dist = dist 
        if min_dist < 10000:
            if min_dist / sec_min_dist < 0.6:
                pair_rec.append([l_p, min_p, min_dist, sec_min_dist])

    match_pair = []
    dist_pair = []
    dist_threshold = 20
    while len(match_pair) < 5:
        remove_idx = []
        for idx, i in enumerate(pair_rec):
            if i[2] < dist_threshold:
                match_pair.append([i[0], i[1]])
                dist_pair.append([i[2], i[3]])
                remove_idx.append(idx)
        remove_idx.reverse()
        for i in remove_idx:
            pair_rec.pop(i) 
        #print(len(match_pair), dist_threshold)
        dist_threshold = dist_threshold * 1.5

    return match_pair, dist_pair 
            
def vis_pair(img_1, img_2, pair, main_pair = None, filename = 'pair.png'):
    img_opt = np.hstack((img_1, img_2))
    for i in pair:
        i1, i2 = i[0], i[1]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(img_opt, (int(i1[1]), int(i1[0])), 5, color, 2)
        cv2.circle(img_opt, (int(i2[1] + img_1.shape[1]), int(i2[0])), 5, color, 2)
        cv2.line(img_opt, (int(i1[1]), int(i1[0])), (int(i2[1] + img_1.shape[1]), int(i2[0])), (0, 255, 255), 3)
    if main_pair != None:
        i1, i2 = main_pair[0], main_pair[1]
        cv2.line(img_opt, (int(i1[1]), int(i1[0])), (int(i2[1] + img_1.shape[1]), int(i2[0])), (255, 255, 0), 3)
    cv2.imwrite(filename, img_opt[:, :, ::-1]) 

def move_amount(pair, dist_pair): # pair: a list
    min_err = 1e10
    for i, dis_i in zip(pair, dist_pair):
        move = np.array([i[0][0] - i[1][0], i[0][1] - i[1][1]]) 
        pair_np = np.array(pair)
        pair_np[:, 1, :] += move
        
        err = pair_np[:, 0, :] - pair_np[:, 1, :]
        err = np.mean(np.sum(err ** 2, axis = 1) ** 0.5)
        if err < min_err:
            min_err = err
            min_move = move 
            main_pair = i
    min_move = min_move.astype('int')
    #print(f"min_move = {min_move}, main_pair = {main_pair}")
    return min_move, main_pair 

def stitch(img_list, move_list, drifted = True):
    if len(img_list) != len(move_list) + 1:
        print(f"Error! two lists' length doesn't match, length = {len(img_list)} {len(move_list)}")
        exit()

    img_opt = img_list[0].astype('float')
    img_shape = img_list[0].shape
    move_cum = np.zeros((2, ), dtype = 'int') # cumulative move
    for img, mv in zip(img_list[1:], move_list):
        img = img.astype('float')
        move_cum += mv
        #print(mv, move_cum)
        if move_cum[0] < 0:
            shape_0 = img_opt.shape[0] + abs(move_cum[0])
        else:
            shape_0 = max(img_opt.shape[0], img.shape[0] + abs(move_cum[0]))
        shape_1 = img.shape[1] + abs(move_cum[1])
        img_tmp = np.zeros((shape_0, shape_1, 3))
        
        #print(img.shape, img_opt.shape, img_tmp.shape)
        
        if mv[0] > 0:
            blend_width = img.shape[1] - mv[1]
            blend_height = img.shape[0] - mv[0]
            img[:blend_height, :blend_width, :] *= weight_0to1(blend_width).reshape(1, -1, 1)
            if move_cum[0] == mv[0]:
                img_opt[mv[0]:mv[0]+blend_height, -blend_width:, :] *= weight_0to1(blend_width, reverse = True).reshape(1, -1, 1)
            else:
                img_opt[-blend_height:, -blend_width:, :] *= weight_0to1(blend_width, reverse = True).reshape(1, -1, 1)
        else:
            blend_width = img.shape[1] - mv[1]
            blend_height = img.shape[0] + mv[0]
            img[-blend_height:, :blend_width, :] *= weight_0to1(blend_width).reshape(1, -1, 1)
            if move_cum[0] == mv[0]:
                img_opt[:blend_height, -blend_width:, :] *= weight_0to1(blend_width, reverse = True).reshape(1, -1, 1)
            else:
                img_opt[mv[0]-blend_height:mv[0], -blend_width:, :] *= weight_0to1(blend_width, reverse = True).reshape(1, -1, 1)

        
        if move_cum[0] < 0:
            img_tmp[-move_cum[0]:-move_cum[0]+img_opt.shape[0], :img_opt.shape[1], :] = img_opt
            img_tmp[:img.shape[0], -img.shape[1]:, :] += img 
            move_cum[0] = 0 
        else:
            img_tmp[:img_opt.shape[0], :img_opt.shape[1], :] = img_opt
            img_tmp[-img.shape[0]:, -img.shape[1]:, :] += img 
        
        img_opt = img_tmp 
    if drifted:
        img_opt = drift(img_opt, move_cum[0], img_opt.shape[1] - move_list[-1][1], img_shape[0])

    return img_opt

def weight_0to1(n, reverse = False):
    if n < 1:
        print('error in weitgh0to1, n =', n)
    if reverse:
        return (np.arange(n, dtype = 'float') / (n - 1))[::-1]
    return np.arange(n, dtype = 'float') / (n - 1)

def drift(img, h_move, r_place, ori_img_height):
    for i in range(img.shape[1]):
        drift_height = int(min((h_move * i / r_place), h_move))
        if drift_height != 0:
            img[:-drift_height, i] = img[drift_height:, i]
    img = img[:ori_img_height - 10] # remove the black space
    return img 
    
        
    
            
if __name__ == '__main__':
    exit()

