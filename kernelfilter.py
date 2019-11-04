import cv2
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


#read image and convert to grayscale **
#resized due to timing
image = cv2.imread('toucan.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image, None,fx=0.1, fy =0.1,interpolation=cv2.INTER_CUBIC)
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image_gray = image_gray.astype('float32') #convert to float 32 because the pixel intensities are more prominent

#load toucan template
template = cv2.imread("toucan_tem.jpg")
template = cv2.cvtColor(template,cv2.COLOR_BGR2RGB)
template = cv2.resize(template, None,fx=0.1, fy =0.1,interpolation=cv2.INTER_CUBIC)
#template = image[ startY:startY+height, startX:startX+width,:]
template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
template_gray = template_gray.astype('float32')


#scale and show image
image.shape
template_gray.shape

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(image_gray)
image_gray.dtype


#displays template

image = cv2.imread('toucan.jpg')


image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image, None,fx=0.1, fy =0.1,interpolation=cv2.INTER_CUBIC)
orig = image.copy()
# orig = messi_im.copy().astype('float32')
# orig-=np.min(orig)
# orig/=np.max(orig)

startX = 150
width = 29
startY = 20
height = 41
template =  cv2.imread('toucan_tem.jpg')
template = cv2.cvtColor(template,cv2.COLOR_BGR2RGB)
template = cv2.resize(template, None,fx=0.1, fy =0.1,interpolation=cv2.INTER_CUBIC)


# template = template.astype('float32')
# template-=np.min(template)
# template/=np.max(template)


image.shape
template.shape

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.imshow(template)
plt.subplot(1,2,2)
plt.imshow(orig)


#takes image height and width and calculates response size

iheight, iwidth = image_gray.shape
theight, twidth = template_gray.shape
response =np.array(image_gray[:(image.shape[0]-theight)+1,:(image.shape[1]-twidth)+1],'float32')
response.shape
response.dtype



# SQUARE DIFFERENCE EQUATION -- SUMMATION OF TEMPLATE PX SUBSTRACTED BY CORRESSPONDING IMAGE PX SQUARED

## shape function returns height, width

#convert to float 32
floatim = image_gray.astype('float32')

floattemp = template_gray.astype('float32')

#create a function that takes the test image and template and returns response with square difference equation in the for loop

def identify_template(image, template, response):
    height , width = response.shape
    theight, twidth = floattemp.shape
    for x in range(height-theight):        #iterates over response
        for y in range(width-twidth):
            for c in range (theight):         #c and d iterate over template
                for d in range (twidth):
                    response[x+c,y+d] += (template[c,d] - image[x+c,y+d]) ** 2
    return response


response_image = identify_template(floatim, floattemp, response)
_=plt.imshow(response_image)

## cv2.match template

#brightest pixel in gray picture should denote matched location

orig = image.copy()
method = cv2.TM_CCOEFF
h, w = template.shape[0:2]

res = cv2.matchTemplate(image, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc

bottom_right = (top_left[0] + w, top_left[1] + h)

matched_im = cv2.rectangle(orig,top_left, bottom_right, 255, 1)
_=plt.figure(figsize=(15,10))
_=plt.subplot(1,2,1)
_=plt.title('matched location')
_=plt.imshow(matched_im)
_=plt.subplot(1,2,2)
_=plt.title('response')
_=plt.imshow(res,plt.cm.gray)



