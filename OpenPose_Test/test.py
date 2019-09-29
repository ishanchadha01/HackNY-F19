import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_image(frame):
    # MPI
    protoFile = "/Users/khanhnguyen/Downloads/pose_deploy_linevec.prototxt"
    weightsFile = "/Users/khanhnguyen/Downloads/pose_iter_160000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile);
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]

    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368

    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)
    output = net.forward()

    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)
    return frame, frameCopy, points


#Read the person image
person = cv2.imread("person1.jpg")

#Get person, personcopy and locaition of the points on the image
person, personCopy, points = get_image(person)


plt.figure(figsize=[10, 6])
plt.imshow(cv2.cvtColor(personCopy, cv2.COLOR_BGR2RGB))
plt.show()

#read the shirt image. Process and isolate the shirt
img_rgb = cv2.imread("shirt.jpeg")
img = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)
img = cv2.bilateralFilter(img,9,105,105)
r,g,b=cv2.split(img)
equalize1= cv2.equalizeHist(r)
equalize2= cv2.equalizeHist(g)
equalize3= cv2.equalizeHist(b)
equalize=cv2.merge((r,g,b))

equalize = cv2.cvtColor(equalize,cv2.COLOR_RGB2GRAY)

ret,thresh_image = cv2.threshold(equalize,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
equalize= cv2.equalizeHist(thresh_image)

canny_image = cv2.Canny(equalize,250,255)
canny_image = cv2.convertScaleAbs(canny_image)
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)

contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
c=contours[0]
final = cv2.drawContours(img, [c], -1, (255,0, 0), 3)

mask = np.zeros(img_rgb.shape,np.uint8)

new_image = cv2.drawContours(mask,[c],0,255,-1,)
new_image = cv2.bitwise_and(img_rgb, img_rgb, mask = equalize)
new_image = cv2.drawContours(mask,[c], -1, (255,255,255), -1)
new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(new_image_gray, 100, 255, cv2.THRESH_BINARY)
final = cv2.bitwise_and(img_rgb, img_rgb, mask = thresh1)
final = cv2.cvtColor(final,cv2.COLOR_BGR2RGB)

#Turn final into an image then resize and save
new_shirt = Image.fromarray(final, 'RGB')
new_shirt = new_shirt.resize((700, 700))
new_shirt.save('test.png')

#Read the new image created
new_shirt = cv2.imread('test.png')

# Create an all white mask
mask = 255 * np.ones(new_shirt.shape, new_shirt.dtype)
# The location of the center of the src in the dst
mid_height = (points[3][0] + points[6][0]) / 2
mid_width = (points[0][1] + points[12][1]) /2
center = (int(mid_height), int(mid_width))

print(mid_width, mid_height)

# Seamlessly clone src into dst and put the results in output
mixed_clone = cv2.seamlessClone(new_shirt, personCopy, mask, center, cv2.MIXED_CLONE)


plt.figure(figsize=[10, 6])
plt.imshow(cv2.cvtColor(mixed_clone, cv2.COLOR_BGR2RGB))
plt.show()
