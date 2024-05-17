import numpy as np
import cv2
import cv2.aruco as aruco

import pickle
import time
import networkx as nx
import matplotlib.pyplot as plt

from utils import rad_to_deg, calculate_net_rotation, get_rotation_difference



ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

class Orientation:
    # list of [roll, pitch, yaw]
    def __init__(self, args):
        self.roll = args[0]
        self.pitch = args[1]
        self.yaw = args[2]
    
    def __repr__(self):
        return str([self.roll, self.pitch, self.yaw])
    
    def value(self):
        return np.linalg.norm([self.roll, self.pitch, self.yaw])

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
        if not self.cap:
            raise Exception('Camera not initialized; call init_camera() first. ')
        
        _, frame = self.cap.read()
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def detect_aruco_markers_in_frame(self, frame):
        if len(frame.shape)>2 and frame.shape[2]>1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
    
    def drawAxes(self, bw_frame):
        # Define the aruco dictionary and charuco board
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, parameters)
        rvec, tvec = None, None
        print(f'type of gray img is {type(bw_frame)}')
        # lists of ids and the corners beloning to each id
        corners, ids, rej = detector.detectMarkers(image=bw_frame)    # ignore rejected points in image
        print(f'number of rejected points is {len(rej)}')
        print(f'rejected points are {rej}')
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # estimate pose of each marker 
                rvec, tvec, markerPoints = self.my_estimatePoseSingleMarkers(corners[i])
                # print(rad_to_deg(rvec))
                aruco.drawDetectedMarkers(bw_frame, corners, borderColor=(255,0,0))  # Draw A square around the markers
                # aruco.DrawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
                rvec = np.array(rvec)
                tvec = np.array(tvec)
                bw_frame = cv2.drawFrameAxes(bw_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec,0.03)
        return bw_frame
    
    
    def annotate_marker_image(self, color_frame, square=True, axes=True):
        frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = self.detect_aruco_markers_in_frame(frame)
        rvecs, tvecs, _ = self.my_estimatePoseSingleMarkers(corners)
        if square:
            aruco.drawDetectedMarkers(color_frame, corners, borderColor=(255,0,0))
        if axes:
            rvecs = np.array(rvecs)
            tvecs = np.array(tvecs)
            frame = self.drawAxes(frame)
            for rvec, tvec in zip(rvecs, tvecs):
                color_frame = cv2.drawFrameAxes(color_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
        return frame
    
    
    def get_marker_orientations(self, color_frame):
        frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        result = self.detect_aruco_markers_in_frame(frame)
        if result is None:
            return []
        corners, ids, rejected_img_points = result
        ids = [x[0] for x in ids]
        rvecs, tvecs, _ = self.my_estimatePoseSingleMarkers(corners)
        return list(zip(ids, rvecs))
        

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
        return f'M_{self.marker_id:02d}'

class ArucoBall:
    def __init__(self):
        self.markers = {}
        self.primaryMarkerID : int = None
        self.network = nx.DiGraph()
        self.all_shortest_paths = None
        
    def add_marker(self, marker: ArucoMarker):
        """
        Add an ArucoMarker object to the graph.
        :param marker: ArucoMarker object to be added
        """
        # self.markers[marker.marker_id] = marker
        self.network.add_node(marker)

    def get_id_list(self):
        return list(self.network.nodes)
        
    def add_marker_adjacency(self, marker_id1: int, marker_id2: int, relative_orientation: list[float]):
        """
        Add an edge between two markers in the graph.
        :param marker_id1: ID of the first marker
        :param marker_id2: ID of the second marker
        :param relative_orientation: Relative orientation between the two markers
        """
        if marker_id1 not in self.network:
            self.add_marker(ArucoMarker(marker_id1, self))
        if marker_id2 not in self.network:
            self.add_marker(ArucoMarker(marker_id2, self))
            
        self.network.add_edge(marker_id1, marker_id2, weight=relative_orientation, label=str(marker_id1)+'-'+str(marker_id2))

    # designate a single marker as the 'primary' marker from which all net orientations are calculated.
    def setPrimaryMarkerID(self, marker_id: int, verbose:bool=False):
        
        if marker_id not in self.network:
            print(f'{marker_id} not in Aruco Marker Network.')
        else:
            self.primaryMarkerID = marker_id
        
        if verbose:print(f'Primary Marker ID set to {marker_id}.')
        
        # store all shortest paths
        self.all_shortest_paths = nx.single_target_shortest_path(self.network, target=self.primaryMarkerID)
        
        # precompute the net rotation matrix from each marker back to the primary marker.
        self.net_rotations_dict = {}
        for startMarker, path in list(self.all_shortest_paths.items()):
            weights = self.get_weights_on_path(path)
            self.net_rotations_dict[startMarker] = calculate_net_rotation(weights)   
        # using this 'net rotation' matrix, given an arbitrary marker orientation, we can calculate the orientation of the primary marker.
        
    
    def get_rotation_to_primary(self, start_marker):
        if start_marker not in self.network.nodes:
            print(f'{start_marker} NOT IN NODES')
            return np.identity(3)
        assert self.primaryMarkerID is not None, 'Primary Marker ID not set.'
        assert start_marker in self.network.nodes, f'Marker ID {start_marker} not in Aruco Marker Network.'
        
        return self.net_rotations_dict[start_marker]
    
    def getPrimaryMarkerOrientation(self, marker_id, marker_rvec):
        
        marker_mat, _ = cv2.Rodrigues(marker_rvec)
        
        rot_difference_to_center = get_rotation_difference(marker_mat, np.identity(3))
        
        rotation_to_primary = self.get_rotation_to_primary(marker_id)
        
        result = rot_difference_to_center @ rotation_to_primary
        
        return result
        return marker_mat @ rotation_to_primary
        return rotation_to_primary
        
    # return the list of nodes in the path from the given marker to the primary marker.
    def getPathToPrimaryMarker(self, marker_id: int):
        assert self.primaryMarkerID is not None, 'Primary Marker ID not set.'
        assert marker_id in self.network.nodes, f'Marker ID {marker_id} not in Aruco Marker Network.'
        return self.all_shortest_paths[marker_id]
        
    def get_marker(self, marker_id):
        """
        Get the ArucoMarker object corresponding to a marker ID.
        :param marker_id: ID of the marker
        :return: ArucoMarker object corresponding to the given ID
        """
        return self.network[marker_id]
    
    # get the list of edge weights (rotation matrices) given a list of nodes to traverse through
    def get_weights_on_path(self, path:list[int], verbose:bool=False):
        if verbose:print(f'path is {path}')
        weights = []
        for i in range(len(path)-1):
            weights.append(self.network[path[i]][path[i+1]]['weight'])
        if verbose:print(f'weights are {weights}')
        return weights

    def getRotationMatrixToPrimary(self, startMarker: int):
        path_to_prim = self.getPathToPrimaryMarker(startMarker)
        w = self.get_weights_on_path(path_to_prim)
        return calculate_net_rotation(w)
    
    def export_graph(self, filename='output/arucoball.gexf'):
        assert filename.endswith('.gexf'), 'Filename must end with .gexf'
        nx.write_gexf(self.network, filename)
        
        
    # add all marker adjacencies to ArucoBall network based on information in a single frame
    def add_adjacencies_from_frame(
        self,
        frame_info:list[list[int], list[float], list[float]],
        verbose=True
        ):
        
        id_list, rvec_list, tvec_list = frame_info
        rvec_list = np.array(rvec_list)
        tvec_list = np.array(tvec_list)
        
        # for each starting node:
        for i, id1 in enumerate(id_list):
            id1=int(id1)
            m1, jac = cv2.Rodrigues(rvec_list[i])
            if id1 not in self.network:
                self.add_marker(id1)
            # for each ending node:
            for j, id2 in enumerate(id_list):
                if id2 not in self.network:
                    self.add_marker(id2)
                id2=int(id2)
                if i == j:
                    continue
                m2, jac = cv2.Rodrigues(rvec_list[j])
                # add edge from id1 to id2
                relative_orientation = get_rotation_difference(m1, m2)
                # relative_orientation = rotation_vector_difference(rvec_list[i], rvec_list[j])
                print(verbose*' '.join(map(str,('adding edge from', id1, 'to', id2, 'with relative orientation', relative_orientation))))
                self.add_marker_adjacency(id1, id2, relative_orientation)
                
                # add edge from id2 to id1
                inverse_relative_orientation = get_rotation_difference(m2, m1)
                # inverse_relative_orientation = rotation_vector_difference(rvec_list[j], rvec_list[i])
                print(verbose*' '.join(map(str,('adding edge from', id2, 'to', id1, 'with relative orientation', inverse_relative_orientation))))
                self.add_marker_adjacency(id2, id1, inverse_relative_orientation)
                
                
    def draw_marker_graph(self, draw_labels=True):
        G = self.network    # nx.DiGraph object
        # Draw the graph
        pos = nx.spring_layout(G)  # Positions of nodes
        nx.draw(G, pos, with_labels=True, node_size=800, node_color="skyblue", font_size=12, font_weight="bold")
        if draw_labels:
            edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        else:
            nx.draw_networkx_edge_labels(G, pos)
        plt.title("Aruco Marker Graph")
        plt.show()
    
    def load(self, filepath):
        f = open(filepath, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict) 

    def save(self, filepath):
        f = open(filepath, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
    
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
    print('nothing to run here.')