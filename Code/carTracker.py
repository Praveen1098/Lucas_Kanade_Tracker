import cv2
import numpy as np
import argparse
import glob

def jacobian(x_shape, y_shape):
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x, y = np.meshgrid(x, y)
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))
    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)
    return jacob

def adjust_gamma(image, gamma=1.0):
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    return cv2.LUT(image, table)

def CLAHE(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    return cl1

def shift_data_Z_score(tmp_mean, img):
    
    tmp_mean_matrix = np.full((img.shape), tmp_mean)
    img_mean_matrix  = np.full((img.shape), np.mean(img))    
    std_ = np.std(img)    
    z_score = np.true_divide((img.astype(int) - tmp_mean_matrix.astype(int)), std_)
    
    dmean = np.mean(img)-tmp_mean
    
    if dmean < 10:
        shifted_img = -(z_score*std_).astype(int) + img_mean_matrix.astype(int)
        
    else:
        shifted_img = (z_score*std_).astype(int) + img_mean_matrix.astype(int)
    
    return shifted_img.astype(dtype = np.uint8)

def InverseLK(img, tmp, parameters, rect, p):
    # Initialization
    rows, cols = tmp.shape

    lr, iteration = parameters
    # Calculate gradient of template
    grad_x = cv2.Sobel(tmp, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(tmp, cv2.CV_64F, 0, 1, ksize=5)

    # Calculate Jacobian
    jacob = jacobian(cols, rows)
    p=np.zeros(6)


    # Set gradient of a pixel into 1 by 2 vector
    grad = np.stack((grad_x, grad_y), axis=2)
    grad = np.expand_dims((grad), axis=2)
    steepest_descents = np.matmul(grad, jacob)
    steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))

    # Compute Hessian matrix
    hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0, 1))
    for _ in range(iteration):
        # Calculate warp image
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        warp_img = cv2.warpAffine(img, warp_mat, (0, 0))
        warp_img = warp_img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        # Compute the error term
        error = tmp.astype(float) - warp_img.astype(float)
        error = error.reshape((rows, cols, 1, 1))
        update = (steepest_descents_trans * error).sum((0, 1))
        # print("uodate", update)
        d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape((-1))

        # Update p
        d_p_deno = (1 + d_p[0]) * (1 + d_p[3]) - d_p[1] * d_p[2]
        d_p_0 = (-d_p[0] - d_p[0] * d_p[3] + d_p[1] * d_p[2]) / d_p_deno
        d_p_1 = (-d_p[1]) / d_p_deno
        d_p_2 = (-d_p[2]) / d_p_deno
        d_p_3 = (-d_p[3] - d_p[0] * d_p[3] + d_p[1] * d_p[2]) / d_p_deno
        d_p_4 = (-d_p[4] - d_p[3] * d_p[4] + d_p[2] * d_p[5]) / d_p_deno
        d_p_5 = (-d_p[5] - d_p[0] * d_p[5] + d_p[1] * d_p[4]) / d_p_deno

        p[0] += lr * (d_p_0 + p[0] * d_p_0 + p[2] * d_p_1)
        p[1] += lr * (d_p_1 + p[1] * d_p_0 + p[3] * d_p_1)
        p[2] += lr * (d_p_2 + p[0] * d_p_2 + p[2] * d_p_3)
        p[3] += lr * (d_p_3 + p[1] * d_p_2 + p[3] * d_p_3)
        p[4] += lr * (d_p_4 + p[0] * d_p_4 + p[2] * d_p_5)
        p[5] += lr * (d_p_5 + p[1] * d_p_4 + p[3] * d_p_5)
    cv2.imshow('Gammacorrect_img', warp_img)
    k=cv2.waitKey(1)
    return p


ROIs={0:(73,53,104,89),390:(225 , 76 , 64 , 51),280:(190   , 63  ,68 , 55),480:(202,70,68,52),595:(228  ,61 , 85 , 65)
  ,170:(110,66 ,81 ,65),209:(139    ,67  ,74  ,55)} # Dataset:(x,y,w,h)
filepath="/home/arjun/Desktop/LKT/Car4/img/0001.jpg"
frame=cv2.imread(filepath)
x,y,w,h=ROIs[0]
color_template = frame[y:y+h,x:x+w]
rect=(x,y,w,h)
refPt=[[x, y], [x+w, y+h]]
template = cv2.cvtColor(color_template, cv2.COLOR_BGR2GRAY)
T = np.float32(template)/255
p=np.zeros(6)
images=[]
for img in glob.glob((r"/home/arjun/Desktop/LKT/Car4/img/*.jpg")):
                images.append(img)
images.sort()
images=images[1:]
parameters = [2, 200]

count=2
for img in images:
        if(count==170 or count ==209 or count==280 
            or count==390 or count ==480 or count ==595) :
            frame=cv2.imread(img)
            if count ==170 or count ==209:
                        frame = adjust_gamma(frame,gamma=1.5)
            if count ==390 or count ==480 or count ==595 :
                         parameters = [2.5, 300]
            x,y,w,h=ROIs[count]
            rect=(x,y,w,h)
            color_template = frame[y:y+h,x:x+w]
            template = cv2.cvtColor(color_template, cv2.COLOR_BGR2GRAY)
            T = np.float32(template)/255
            refPt=[[x, y], [x+w, y+h]]
            p=np.zeros(6)        
        color_frame=cv2.imread(img)
        if 183 <count <230:
                        color_frame = adjust_gamma(color_frame,gamma=1.5)
        gray_frame=cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        I = np.float32(gray_frame)/255
        p = InverseLK(I, T, parameters, refPt, p)
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        warp_mat = cv2.invertAffineTransform(warp_mat)
        rectangle=[[rect[0],rect[1]],[rect[0]+w,rect[1]],[rect[0]+w,rect[1]+h],[rect[0],rect[1]+h]]
        box=np.array(rectangle)
        box=box.T
        box=np.vstack((box,np.ones((1,4))))
        pts=np.dot(warp_mat,box)
        pts=pts.T
        pts=pts.astype(np.int32)
        print(count)

        count+=1
        cv2.polylines(color_frame, [pts], True, (0,255,255), 2)
        cv2.imshow('Tracked Image',color_frame)
        cv2.waitKey(1)
        

