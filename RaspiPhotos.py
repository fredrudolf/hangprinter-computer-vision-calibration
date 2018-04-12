from enum import Enum
import os
import cv2

class Res(Enum):
    LOW = (320, 240)
    MP3 = (2048, 1536)
    MP5 = (2592, 1944)
    MP8 = (3280, 2464)

def capture(filename, res):
    "System command for raspistill capture"
    print("Capture {} with: {}x{}".format(filename, *res.value))
    cmd = "raspistill -w {} -h {} -o {}".format(*res.value, filename)
    os.system(cmd)

def view(filename, res):
    "Read and view image"
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    cv2.imshow('window', img)
    print('Previewing')
    return cv2.waitKey(0)

def main():

    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('window', (1280,960))
    
    i = 0
    while True:
        # Capture preview
        capture('temp.jpg', Res.LOW)
        key = view('temp.jpg', Res.LOW)
        # Kill program
        if key == ord("q"):
            break
        # Capture all resolutions
        elif key == ord("s"):
            for res in Res:
                filename = 'img{}_{}.jpg'.format(i, res.name)
                print("Saving", filename)
                capture(filename, res)
            i += 1

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

