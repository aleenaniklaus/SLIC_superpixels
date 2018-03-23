import numpy
import sys
import cv2
import tqdm

# using algorithm in 3.2 apply image gradients as computed in eq2:
# G(x,y) = ||I(x+1,y) - I(x-1,y)||^2+ ||I(x,y+1) - I(x,y-1)||^2

def generatePixels():
    indnp = numpy.mgrid[0:SLIC_height,0:SLIC_width].swapaxes(0,2).swapaxes(0,1)
    for i in tqdm.tqdm(range(SLIC_ITERATIONS)):
        SLIC_distances = SLIC_FLT_MAX * numpy.ones(SLIC_img.shape[:2])
        for j in range(SLIC_centers.shape[0]):
            xlow, xhigh = int(SLIC_centers[j][3] - SLIC_step), int(SLIC_centers[j][3] + SLIC_step)
            ylow, yhigh = int(SLIC_centers[j][4] - SLIC_step), int(SLIC_centers[j][4] + SLIC_step)

            if xlow <= 0:
                xlow = 0
            if xhigh > SLIC_width:
                xhigh = SLIC_width
            if ylow <=0:
                ylow = 0
            if yhigh > SLIC_height:
                yhigh = SLIC_height

            cropimg = SLIC_labimg[ylow : yhigh , xlow : xhigh]
            colordiff = cropimg - SLIC_labimg[int(SLIC_centers[j][4]), int(SLIC_centers[j][3])]
            colorDist = numpy.sqrt(numpy.sum(numpy.square(colordiff), axis=2))

            yy, xx = numpy.ogrid[ylow : yhigh, xlow : xhigh]
            pixdist = ((yy-SLIC_centers[j][4])**2 + (xx-SLIC_centers[j][3])**2)**0.5
            dist = ((colorDist/SLIC_nc)**2 + (pixdist/SLIC_ns)**2)**0.5

            distanceCrop = SLIC_distances[ylow : yhigh, xlow : xhigh]
            idx = dist < distanceCrop
            distanceCrop[idx] = dist[idx]
            SLIC_distances[ylow : yhigh, xlow : xhigh] = distanceCrop
            SLIC_clusters[ylow : yhigh, xlow : xhigh][idx] = j

        for k in range(len(SLIC_centers)):
            idx = (SLIC_clusters == k)
            colornp = SLIC_labimg[idx]
            distnp = indnp[idx]
            SLIC_centers[k][0:3] = numpy.sum(colornp, axis=0)
            sumy, sumx = numpy.sum(distnp, axis=0)
            SLIC_centers[k][3:] = sumx, sumy
            SLIC_centers[k] /= numpy.sum(idx)

# At the end of the process, some stray labels may remain meaning some pixels
# may end up having the same label as a larger pixel but not be connected to it
# In the SLIC paper, it notes that these cases are rare, however this 
# implementation seems to have a lot of strays depending on the inputs given

def createConnectivity():
    label = 0
    adjlabel = 0
    lims = int(SLIC_width * SLIC_height / SLIC_centers.shape[0])
    
    new_clusters = -1 * numpy.ones(SLIC_img.shape[:2]).astype(numpy.int64)
    elements = []
    for i in range(SLIC_width):
        for j in range(SLIC_height):
            if new_clusters[j, i] == -1:
                elements = []
                elements.append((j, i))
                for dx, dy in [(-1,0), (0,-1), (1,0), (0,1)]:
                    x = elements[0][1] + dx
                    y = elements[0][0] + dy
                    if (x>=0 and x < SLIC_width and 
                        y>=0 and y < SLIC_height and 
                        new_clusters[y, x] >=0):
                        adjlabel = new_clusters[y, x]
            count = 1
            c = 0
            while c < count:
                for dx, dy in [(-1,0), (0,-1), (1,0), (0,1)]:
                    x = elements[c][1] + dx
                    y = elements[c][0] + dy

                    if (x>=0 and x<SLIC_width and y>=0 and y<SLIC_height):
                        if new_clusters[y, x] == -1 and SLIC_clusters[j, i] == SLIC_clusters[y, x]:
                            elements.append((y, x))
                            new_clusters[y, x] = label
                            count+=1
                c+=1
            if (count <= lims >> 2):
                for c in range(count):
                    new_clusters[elements[c]] = adjlabel
                label-=1
            label+=1
    SLIC_new_clusters = new_clusters

def displayContours(color):

    isTaken = numpy.zeros(SLIC_img.shape[:2], numpy.bool)
    contours = []

    for i in range(SLIC_width):
        for j in range(SLIC_height):
            nr_p = 0
            for dx, dy in [(-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1)]:
                x = i + dx
                y = j + dy
                if x>=0 and x < SLIC_width and y>=0 and y < SLIC_height:
                    if isTaken[y, x] == False and SLIC_clusters[j, i] != SLIC_clusters[y, x]:
                        nr_p += 1

            if nr_p >= 2:
                isTaken[j, i] = True
                contours.append([j, i])

    for i in range(len(contours)):
        SLIC_img[contours[i][0], contours[i][1]] = color

def findLocalMinimum(center):
    min_grad = SLIC_FLT_MAX
    loc_min = center
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = SLIC_labimg[j+1, i]
            c2 = SLIC_labimg[j, i+1]
            c3 = SLIC_labimg[j, i]
            if ((c1[0] - c3[0])**2)**0.5 + ((c2[0] - c3[0])**2)**0.5 < min_grad:
                min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                loc_min = [i, j]
    return loc_min

img = cv2.imread(sys.argv[1])
nr_superpixels = int(sys.argv[2])
step = int((img.shape[0]*img.shape[1]/nr_superpixels)**0.5)

SLIC_step = step
SLIC_nc = int(sys.argv[3])
SLIC_ns = step
SLIC_FLT_MAX = 1
SLIC_ITERATIONS = 6
SLIC_img = img
SLIC_height, SLIC_width = img.shape[:2]
SLIC_labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(numpy.float64)
SLIC_clusters = -1 * numpy.ones(SLIC_img.shape[:2])
SLIC_distances = SLIC_FLT_MAX * numpy.ones(SLIC_img.shape[:2])

centers = []
for i in range(SLIC_step, SLIC_width - int(SLIC_step/2), SLIC_step):
    for j in range(SLIC_step, SLIC_height - int(SLIC_step/2), SLIC_step):
        
        nc = findLocalMinimum(center=(i, j))
        color = SLIC_labimg[nc[1], nc[0]]
        center = [color[0], color[1], color[2], nc[0], nc[1]]
        centers.append(center)
SLIC_center_counts = numpy.zeros(len(centers))
SLIC_centers = numpy.array(centers)

generatePixels()
createConnectivity()
displayContours([0.0, 0.0, 0.0])
cv2.imshow("superpixels", SLIC_img)
cv2.waitKey(0)
cv2.imwrite("SLICimg.jpg", SLIC_img)