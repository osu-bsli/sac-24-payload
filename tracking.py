import numpy as np
import cv2
import cv2.aruco as aruco
import skimage.io as io

cap = cv2.VideoCapture(0)  # Get the camera source

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# https://stackoverflow.com/questions/75750177/solve-pnp-or-estimate-pose-single-markers-which-is-better
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
    corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


def rad_to_deg(rad_list):
    return list(map(lambda x: x * 180 / np.pi, rad_list))


def track(matrix_coefficients, distortion_coefficients):
    while True:
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale

        # Define the aruco dictionary and charuco board
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, parameters)

        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = detector.detectMarkers(image=gray)
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = my_estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
                # print('\t'.join(str(x) for x in rvec))
                print(rad_to_deg(rvec))
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                # aruco.DrawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
                rvec = np.array(rvec)
                tvec = np.array(tvec)
                frame = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec,0.03)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    # Load calibration data
    camera_matrix = np.load('cam_properties/camera_matrix.npy')
    dist_coeffs = np.load('cam_properties/dist_coeffs.npy')

    track(camera_matrix, dist_coeffs)