from PIL import Image
import cv2
def jpeg_to_png(img_path='images/cap.jpeg', png_path=''):
    png_path=img_path.split('.')[0]+'.png'
    im = Image.open(img_path)
    im.save(png_path)
    return png_path

def read_frame(png_path='images/cap.png'):
    frame = cv2.imread(png_path,0)
    return frame

def show_frame(frame):
    cv2.imshow('cap',frame)
    cv2.waitKey(0)

if __name__=='__main__':
    print('<UTILS.PY>')