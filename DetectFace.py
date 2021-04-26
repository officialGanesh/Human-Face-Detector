# Import the required modules
import cv2 as cv
from random import randrange

# grabbing the data
face_detect_data = cv.CascadeClassifier('DataSet.xml')


def Detect_faces():
    """This function help us to detect faces on the given Image by converting the color image into grayScale"""

    try:
        # actual image
        img = cv.imread('Images\Elon.jpg')
        cv.imshow("Elon-Face",img)

        # grayScale images
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        cv.imshow("Gray-Face-Elon",gray)

        # detecting the faces

        face_coordinates = face_detect_data.detectMultiScale(gray)
        #print(len(face_coordinates))
        # print(face_coordinates)

    except Exception as e:
        print("Something went wrong --> ",e)


if __name__ == "__main__":
    Detect_faces()

    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Code Completed 🔥")
