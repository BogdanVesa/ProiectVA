import cv2
import numpy as np

scale_ratio = 1

def maskCreate(img):

    converted = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # white color mask
    lower = np.uint8([0, 210,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 20,   20, 100])
    upper = np.uint8([ 30, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(img, img, mask = white_mask)

def edgeDetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gauss = cv2.GaussianBlur(gray, (9,9), 0)
    canny = cv2.Canny(gauss, 50, 150)
    return canny

def filterRegion(img, vertices):
    mask = np.zeros_like(img)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(img, mask)
def selectRegion(img):
    rows, cols = img.shape[:2]
    global bottomLeft
    global topLeft
    global bottomRight
    global topRight
    bottomLeft  = [int(cols*0.2), int(rows*0.7)]
    topLeft     = [int(cols*0.5), int(rows*0.5)]
    bottomRight = [int(cols*0.65), int(rows*0.7)]
    topRight    = [int(cols*0.55), int(rows*0.5)]

    vertices = np.array([[bottomLeft, topLeft, topRight, bottomRight]], dtype=np.int32)
    result = filterRegion(img, vertices)
    
    return result

    
def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/90, threshold=20, minLineLength=20, maxLineGap=300)


def averageLine(lines):
  line = lines[0][0]
  minX = line[0]
  minY = line[1]
  maxX = line[2]
  maxY = line[3]
  for i in range(1, len(lines)):
    l = lines[i][0]
    if minX > l[0]:
      minX = l[0]
    if minX > l[2]:
      minX = l[2]
    if minY > l[1]:
      minY = l[1]
    if maxX < l[2]:
      maxX = l[2]
    if maxX < l[0]:
      maxX = l[0]
    if maxY < l[3]:
      maxY = l[3]
  
  startPoint = []
  midPoint = []
  endPoint = []
  startPoint.append(minX+ abs(maxX - minX)//2)
  startPoint.append(minY)
  endPoint.append(minX+abs(maxX - minX)//2)
  endPoint.append(maxY)
  midPoint.append(minX + abs(maxX-minX)//2)
  midPoint.append(minY + abs(maxY-minY)//2)

  return startPoint, endPoint
def sections(region, sections):
  distLeft=[]
  distLeft.append(abs(bottomLeft[0] - topLeft[0])//sections)
  distLeft.append(abs(bottomLeft[1] - topLeft[1])//sections)
  distRight =[]
  distRight.append(abs(bottomRight[0]- topRight[0])//sections)
  distRight.append(abs(bottomRight[1]- topLeft[1])//sections)
  newLines = []
  currentPos1 = bottomLeft
  currentPos2 = bottomRight
  while (currentPos1[0]< topLeft[0] and currentPos1[1]>topLeft[1] 
        and currentPos2[0] > topRight[0] and currentPos2[1] > topRight[1]):
    newLeft = [ currentPos1[0]+ distLeft[0], currentPos1[1]-distLeft[1]]
    newRight = [ currentPos2[0]- distLeft[0], currentPos2[1]-distLeft[1]]
    newVertices= np.array([[currentPos1, newLeft, newRight, currentPos2]], dtype=np.int32)
    filter = filterRegion(region, newVertices)
    lines = hough_lines(filter)
    newLine = averageLine(lines)
    newLines.append(newLine)
    currentPos1 = newLeft
    currentPos2 = newRight

  return newLines

def main():
    cap = cv2.VideoCapture("driving3.mp4")

    while cap.isOpened():

        success, frame = cap.read()

        if success:
            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            frame = cv2.resize(frame, (width, height))
            copy = frame.copy()
            mask = maskCreate(frame)
            edge = edgeDetection(mask)
            region = selectRegion(edge)
            newLines = sections(region,5)
            for line in newLines:
              start = line [0]
              end = line [1]
              cv2.line(frame, start, end, (0,0,255),3)

            cv2.imshow("copy", copy)
            cv2.imshow("Mask", mask)
            cv2.imshow("edge",edge)
            cv2.imshow("region", region)
            cv2.imshow("End result", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
		
        else:
            break
    cap.release()


if __name__ == "__main__":
    main()