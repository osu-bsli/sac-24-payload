import numpy as np
import cv2
import cv2.aruco as aruco

import time
import networkx as nx
import matplotlib.pyplot as plt

from utils import rad_to_deg

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

class Orientation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __repr__(self):
        return f"[Roll: {self.roll}, Pitch: {self.pitch}, Yaw: {self.yaw}]"
    
    def __repr__(self):
        return str([self.roll, self.pitch, self.yaw])

class ArucoVision:
    def __init__(self) -> None:
        self.cap = None
        # camera params
        self.camera_matrix = np.load('cam_properties/camera_matrix.npy')    # camera matrix
        self.dist_coeffs = np.load('cam_properties/dist_coeffs.npy')    # distortion coefficients
        # aruco detector
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(ARUCO_DICT, self.parameters)
        self.marker_size = 0.02
    
    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        time.sleep(2)
        
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def get_frame(self):
        _, frame = self.cap.read()
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def detect_aruco_markers_in_frame(self, frame):
        corners, ids, rejected_img_points = self.detector.detectMarkers(image=frame)
        
        if ids is not None:
            return corners, ids, rejected_img_points
        else:
            return None
    
    def get_marker_orientation(self, marker_corner_locations, verbose=False):
        rvec, tvec, _ = self.my_estimatePoseSingleMarkers(marker_corner_locations)
        angles = rad_to_deg(rvec) # convert to degrees
        if verbose:
            print(f'type of angles is {type(angles)}')
            print(f'type of first element in angles is {type(angles[0])}')
            print(f'orientation of marker: ',angles)
        if len(angles)>0:
            return rvec, tvec
        else:
            return None
        
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
        
        - The tvec of a marker is the translation (x,y,z) of the marker from the origin
        - The rvec of a marker is a 3D rotation vector which defines both an axis of rotation and the rotation angle about that axis, and gives the marker's orientation. It can be converted to a 3x3 rotation matrix using the Rodrigues function (cv::Rodrigues())
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
    
    def annotate_marker_image(self, frame, corners, rvec, tvec, square=True, axes=True):
        if square:
            aruco.drawDetectedMarkers(frame, corners)
        if axes:
            rvec = np.array(rvec)
            tvec = np.array(tvec)
            frame = cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, length=0.03)
            

class ArucoBall:
    def __init__(self):
        self.markerNetwork = {}
        
    def add_marker(self, marker):
        """
        Add an ArucoMarker object to the graph.
        :param marker: ArucoMarker object to be added
        """
        self.markers[marker.marker_id] = marker
        
    def get_id_list(self):
        return list(self.markers.keys())

    def add_edge(self, marker_id1, marker_id2, relative_orientation):
        """
        Add an edge between two markers in the graph.
        :param marker_id1: ID of the first marker
        :param marker_id2: ID of the second marker
        :param relative_orientation: Relative orientation between the two markers
        """
        marker1 = self.markers.get(marker_id1)
        marker2 = self.markers.get(marker_id2)
        if marker1 and marker2:
            marker1.add_adjacent_marker(marker2, relative_orientation)
            marker2.add_adjacent_marker(marker1, relative_orientation)

    def get_marker(self, marker_id):
        """
        Get the ArucoMarker object corresponding to a marker ID.
        :param marker_id: ID of the marker
        :return: ArucoMarker object corresponding to the given ID
        """
        return self.markers.get(marker_id, None)

class ArucoMarker:
    def __init__(self, marker_id, parentGraph):
        self.marker_id = marker_id
        # dict to store adjacent markers with their relative orientations (this is always Orientation(0,0,0))
        self.adjacent_markers = {}  
        self.parentGraph = parentGraph

    def add_adjacent_marker(self, adjacent_marker, relative_orientation):
        """
        Add an adjacent marker with its relative orientation.
        :param adjacent_marker: ArucoMarker object representing the adjacent marker
        :param relative_orientation: Relative orientation of the adjacent marker with respect to this marker
        """
        self.adjacent_markers[adjacent_marker] = relative_orientation
    
    def get_relative_orientation(self, adjacent_marker):
        """
        Get the relative orientation of an adjacent marker.
        :param adjacent_marker: ArucoMarker object representing the adjacent marker
        :return: Relative orientation of the adjacent marker with respect to this marker
        """
        return self.adjacent_markers.get(adjacent_marker, None)
    
    def __repr__(self):
        return f'M_{self.marker_id:02d}: '+str(self.adjacent_markers)


def draw_aruco_graph(graph):
    G = nx.Graph()

    # Add nodes to the graph
    for marker_id, marker in graph.markers.items():
        G.add_node(marker_id, label=f"Marker {marker_id}")

    # Add edges to the graph
    for marker_id, marker in graph.markers.items():
        for adjacent_marker, relative_orientation in marker.adjacent_markers.items():
            G.add_edge(marker_id, adjacent_marker.marker_id, label=str(relative_orientation))

    # Draw the graph
    pos = nx.spring_layout(G)  # Positions of nodes
    nx.draw(G, pos, with_labels=True, node_size=800, node_color="skyblue", font_size=12, font_weight="bold")
    edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Aruco Marker Graph")
    plt.show()
    
def debug_tracking_loop():
    
    aruco_vision = ArucoVision()
    aruco_vision.init_camera()
    while True:
        frame, bw_frame = aruco_vision.get_frame()
        markers = aruco_vision.detect_aruco_markers_in_frame(bw_frame)
        
        if markers is None:
            print(f'No markers found in frame.')
        else:
            all_corner_lists, marker_ids, rejected_img_points = markers
            
            for corner_list, id in zip(all_corner_lists, marker_ids):
                rvec, tvec = aruco_vision.get_marker_orientation(corner_list)
                print(f'MARKER {id}')
                print(f'orientation of marker {id}:', rvec)
                print(f'translation:', tvec)
                aruco_vision.annotate_marker_image(frame, all_corner_lists, rvec, tvec)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break

def get_markers_in_single_frame(frame, show_annotated = False):
    aruco_vision = ArucoVision()
    bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markers = aruco_vision.detect_aruco_markers_in_frame(bw_frame)
    id_list = []
    rvec_list = []
    tvec_list = []
    if markers is None:
        print(f'No markers found in frame.')
    else:
        all_corner_lists, marker_ids, rejected_img_points = markers
        for corner_list, id in zip(all_corner_lists, marker_ids):
            rvec, tvec = aruco_vision.get_marker_orientation(corner_list)
            # print(f'MARKER {id}')
            # print(f'orientation of marker {id}:', rad_to_deg(rvec[0]))
            # print(f'translation:', tvec)
            if show_annotated:
                aruco_vision.annotate_marker_image(frame, all_corner_lists, rvec, tvec)
            id_list.append(id)
            rvec_list.append(rvec)
            tvec_list.append(tvec)
    if show_annotated:
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
    return id_list, rvec_list, tvec_list

if __name__=='__main__':
    # debug_tracking_loop()
    # read image from file:
    
    # img = cv2.imread('images/test.jpg')
    # id_list, rvec_list, tvec_list = get_markers_in_single_frame(img)
    # print(f'first rvec is {rvec_list[0]}')
    
    
    # print(f'num markers found = {len(rvec_list)}\nRotation vectors:')
    # for id, rot, trans in zip(id_list, rvec_list, tvec_list):
    #     print(f'rot of {id}: {(''.join(map(lambda x:str(x),rot))).replace('\n', '\t')}')
    print(1)