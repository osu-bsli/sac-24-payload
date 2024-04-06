import numpy as np
import cv2
import cv2.aruco as aruco

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
from utils import rad_to_deg


import math


def track(cap, matrix_coefficients, distortion_coefficients):
    # Define the aruco dictionary and charuco board
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, parameters)
    
    while True:
        _, frame = cap.read()
        
        # change frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = detector.detectMarkers(image=frame)
        
        # if there are markers found by detector:
        if np.all(ids is not None):  
            
            # iterate over each marker
            for i in range(0, len(ids)):  
                # Estimate pose of each marker and return the values rvec and tvec -- different from camera coefficients
                rvec, tvec, _ = my_estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
                # print('\t'.join(str(x) for x in rvec))
                print(type(rvec))
                angles = rad_to_deg(rvec)
                print(f'MARKER {ids[i]}')
                print(f'type of angles is {type(angles)}')
                print(f'type of first element in angles is {type(angles[0])}')
                print(f'orientation of marker {ids[i]}:',angles)
                
                # Draw a square around the markers
                aruco.drawDetectedMarkers(frame, corners)  
                
                # Draw Axis
                rvec = np.array(rvec)
                tvec = np.array(tvec)
                frame = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec,0.03)
                
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
        if np.all(ids is not None):
            print('\n\n')
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# https://stackoverflow.com/questions/75750177
def my_estimatePoseSingleMarkers(self, corner_locations):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
    corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    
    marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                [self.marker_size / 2, self.marker_size / 2, 0],
                                [self.marker_size / 2, -self.marker_size / 2, 0],
                                [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corner_locations:
        nada, R, t = cv2.solvePnP(marker_points, c, self.camera_matrix, self.dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def main():
    cap = cv2.VideoCapture(0)  # Get the camera source
    # Load calibration data
    camera_matrix = np.load('cam_properties/camera_matrix.npy')
    dist_coeffs = np.load('cam_properties/dist_coeffs.npy')
    track(cap, camera_matrix, dist_coeffs)


if __name__=="__main__":
    main()