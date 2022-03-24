"""
A set of functions and scripts to demonstrate camera calibration and
registration.

Notes:
  * Parts of this module uses opencv calibration described here:
    https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html
  * and the Chruco board described here:
   https://docs.opencv.org/4.0.0/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html

Copyright (C) Microsoft Corporation. All rights reserved.
"""

# Standard imports.
import os
import fnmatch
import json
import math
import time
import warnings

from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Tuple

# 3rd party imports.
import cv2
import cv2.aruco as aruco
from cv2 import aruco_CharucoBoard
from cv2 import aruco_DetectorParameters as detect_params
import numpy as np

#-------------------------------------------------------------------------------
class CalibrationError(Exception):
  """Error during calibration.
  """

class RegistrationError(Exception):
  """Error during registration.
  """

#-------------------------------------------------------------------------------
@dataclass
class RTMatrix():
  """dataclass for containing rotation and translation between coordinates"""

  rotation:List
  translation:List


#-------------------------------------------------------------------------------
def write_json(json_file:str, data:dict)-> None:
  """Helper function for writing out json files.

  Args:
    json_file (str): full path
    data (dict): Blob of data to write.
  """
  with open(json_file, "w") as j:
    json.dump(data, j, indent=4)

#-------------------------------------------------------------------------------
def r_as_matrix(rotation:np.array):
  """Convert a 3vec rotation array to a rotation matrix.

  Args:
    rotation (np.array): 3 vector array representing rotation.

  Returns:
    [np.array]: Rotation matrix.
  """
  rmat = np.zeros(shape=(3,3))
  cv2.Rodrigues(rotation, rmat)
  return rmat

#-------------------------------------------------------------------------------
def read_opencv_calfile(calfile:str) -> Tuple[np.ndarray,
                                              np.ndarray,
                                              np.ndarray]:
  """Read a calibration file generated by opencv.

  Args:
    calfile (str): full path to an opencv calibration yaml file.

  Raises:
    ValueError: Function will explicitly fail when the cv2 calibration file
      fails to open.

  Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
      k_matrix: The camera intrinsics matrix.
      dist: the distortion matrix.
      img_size: Image size as ndarray objects, in the order width, height.
  """
  fs_calib = cv2.FileStorage(calfile, cv2.FILE_STORAGE_READ)
  if not fs_calib.isOpened():
    raise ValueError(f"failed to open {fs_calib}")

  k_matrix = fs_calib.getNode("K").mat()
  dist = fs_calib.getNode("dist").mat()
  img_size = fs_calib.getNode("img_size").mat()
  fs_calib.release()
  return k_matrix, dist, img_size

#-------------------------------------------------------------------------------
def write_opencv_calfile(calfile:str,
                         k_matrix:np.array,
                         dist8:List,
                         img_size:np.array) -> None:
  """Write out an opencv calibration file for a single camera.

  Args:
    calfile (str): Full path of calibration file.
    k_matrix (np.array): the camera intrinsics matrix.
    dist8 (List): The distortion coefficients.
    img_size (np.array): 2d array of the image size, in the order width, height.
  """
  fs_calib = cv2.FileStorage(calfile, cv2.FILE_STORAGE_WRITE)
  fs_calib.write("K", k_matrix)
  fs_calib.write("dist", dist8)
  fs_calib.write("img_size", img_size)
  fs_calib.release()

#-------------------------------------------------------------------------------
def write_calibration_blob(calibrations:List[str],
               rmat_b_to_a:np.array,
               tvec_b_to_a:np.array,
               out_dir:str):
  """Write all calibration and registration values to a json blob.

  Args:
    calibrations (List[str]): All calibration files.
    rmat_b_to_a (np.array): Rotation matrix array.
    tvec_b_to_a (np.array): Translation vector array.
    out_dir (str): Ouput directory to place the calibration_blob.json file.
  """

  blob = {"CalibrationInformation":{"Cameras":[]}}

  for idx, calibration_file in enumerate(calibrations):
    if idx == 0:
      # RT for the camera used as the origin for all others.
      reshape_r = [1,0,0,0,1,0,0,0,1]
      reshape_t = [0,0,0]
    else:
      reshape_r = rmat_b_to_a.reshape(9, 1).squeeze(1).tolist()
      reshape_t = tvec_b_to_a.squeeze(1).tolist()

    camera_matrix, dist, img_size = read_opencv_calfile(calibration_file)
    intrinsics = [camera_matrix[0][2]/img_size[0][0], #Px
            camera_matrix[1][2]/img_size[1][0], #Py
            camera_matrix[0][0]/img_size[0][0], #Fx
            camera_matrix[1][1]/img_size[1][0], #Fy
            dist[0][0], #K1
            dist[1][0], #K2
            dist[4][0], #K3
            dist[5][0], #K4
            dist[6][0], #K5
            dist[7][0], #K6
            0, #Cx always Zero. (BrownConrady)
            0, #Cy always Zero. (BrownConrady)
            dist[3][0], #P2/Tx
            dist[2][0]] #P1/Ty

    model_type = "CALIBRATION_LensDistortionModelBrownConrady"
    intrinsics_data = {"ModelParameterCount":len(intrinsics),
               "ModelParameters":intrinsics,
               "ModelType":model_type}

    extrinsics = {"Rotation":reshape_r, "Translation":reshape_t}
    calibration = {"Intrinsics":intrinsics_data,
            "Rt":extrinsics,
            "SensorHeight":img_size[1].tolist(),
            "SensorWidth":img_size[0].tolist()}

    blob["CalibrationInformation"]["Cameras"].append(calibration)

  os.makedirs(out_dir, exist_ok=True)
  json_file = os.path.join(out_dir, "calibration_blob.json")
  write_json(json_file, blob)

#-------------------------------------------------------------------------------
def read_board_parameters(json_file: str) -> Tuple[Dict, aruco_CharucoBoard]:
  """Read charuco board from a json file.

  Args:
    json_file (str): fullpath of the board json_file.

  Returns:
    Tuple[dict, aruco_CharucoBoard]:
      target: Target data from json_file.
      board: A single charuco board object.
  """

  with open(json_file) as j_file:
    targets = json.load(j_file)

  target = targets["shapes"][0]
  aruco_dict = aruco.Dictionary_get(target["aruco_dict_name"])
  board = aruco.CharucoBoard_create(target["squares_x"],
                    target["squares_y"],
                    target["square_length"]/1000,
                    target["marker_length"]/1000,
                    aruco_dict)

  return target, board

#-------------------------------------------------------------------------------
def get_image_points(board:aruco_CharucoBoard,
                     marker_ids:np.ndarray) -> np.ndarray:
  """
  Generate markers 3d and 2d positions like getBoardObjectAndImagePoints but for
    Charuco Parameters.

  Args:
    board (aruco_CharucoBoard): A board object from opencv.
    marker_ids (np.ndarray): List of detected charuco marker Ids.

  Returns:
    np.ndarray: numpy array (n*1*3) markers 3d positions.
  """

  object_points = board.chessboardCorners[marker_ids, :]
  return object_points

#-------------------------------------------------------------------------------
def detect_markers(img: np.ndarray,
                   template: str,
                   params:detect_params = None) -> Tuple[List[np.ndarray],
                                                         List[np.ndarray],
                                                         aruco_CharucoBoard]:
  """Detect board markers.

  Args:
    img (np.ndarray): Board image.
    template (str): fullpath of the board json_file.
    params (aruco_DetectorParameters, optional): a cv2 object
      aruco_DetectorParameters. Defaults to None.

  Returns:
    Tuple[List[np.ndarray], List[np.ndarray], aruco_CharucoBoard]:
      charuco_corners: List of detected charuco marker corners.
      charuco_ids: List of detected charuco marker Ids.
      board: charucoboard object.
  """
  # detect markers
  _, board = read_board_parameters(template)

  if params is None:
    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE

  aruco_corners, aruco_ids, _ = aruco.detectMarkers(img,
                            board.dictionary,
                            None,
                            None,
                            params)
  if len(aruco_corners) > 0:
    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                                      aruco_corners,
                                      aruco_ids,
                                      img,
                                      board)
    if charuco_corners is None:
      charuco_corners = []
      charuco_ids = []
      warnings.warn("No charuco corners detected in image.")
  else:
    charuco_corners = []
    charuco_ids = []
    warnings.warn("No charuco corners detected in image.")

  return charuco_corners, charuco_ids, board

#-------------------------------------------------------------------------------
def detect_markers_many_images(imgnames:List[str], template: str):
  """
  Run detect_markers on a large set of png or jpeg images in a single directory,
  with the assumption that all images are viewing the same board.

  Args:
    imgnames (List[str]):Full path to images.
    template (str): Template file json of the board.

  Raises:
    CalibrationError: Not all image sizes are equal.
    CalibrationError: Insufficient number of markers detected. Inspect images
      for poor quality.

  Returns:
    [Tuple]: [List[List[np.ndarray]],
          List[List[np.ndarray]],
          List[List[np.ndarray]],
          Tuple[np.array, np.array],
          aruco_CharucoBoard]
      ccorners_all: All chauco corners detected in every image.
      cids_all: All charuco marker ids detected in every image.
      p3d: Image points.
      img_size: [width; height].
      board: Charucoboard object.
  """
  ccorners_all = []
  cids_all = []
  p3d = []
  img_sizes_all = []

  for imgfile in imgnames:
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    if img is not None:
      ccorners, cids, board = detect_markers(img, template)

      if len(ccorners) > 3:
        ccorners_all.append(ccorners)
        cids_all.append(cids)
        m3d = get_image_points(board, cids)
        p3d.append(m3d)
        sizes = np.array([img.shape[1], img.shape[0]])
        img_sizes_all.append(sizes)

  # check all images sizes are identical.
  rows_equal = [elem[0]==img_sizes_all[0][0] for elem in img_sizes_all]
  cols_equal = [elem[1]==img_sizes_all[0][1] for elem in img_sizes_all]

  if not all(rows_equal) or not all(cols_equal):
    raise CalibrationError("Not all image sizes in data set are the same.")

  img_size = (img_sizes_all[0][0], img_sizes_all[0][1])
  return ccorners_all, cids_all, p3d, img_size, board

#-------------------------------------------------------------------------------
def estimate_pose(img: np.array,
          template: str,
          opencv_calfile: str) -> Tuple[bool,
                        np.ndarray,
                        np.ndarray]:
  """Estimate camera pose using board.

  Args:
    img (np.array): Board image.
    template (str): fullpath of the board json_file.
    opencv_calfile (str): fullpath of the opencv cal file.

  Raises:
    ValueError: Throw an error if the calibration file fails to load.

  Returns:
    Tuple[bool, np.ndarray, np.ndarray]: Returns success of calibration and
      extrinsics.
      retval: Return True of the optimizer converged.
      rvec: rotation array
      tvec: translation array 1*3
  """

  k_matrix, dist, _ = read_opencv_calfile(opencv_calfile)

  rvec = np.full((1, 3), 0.01)
  tvec = np.full((1, 3), 0.01)

  charuco_corners, charuco_ids, board = detect_markers(img, template)
  if len(charuco_corners) > 0:
    retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners,
                              charuco_ids,
                              board,
                              k_matrix,
                              dist,
                              rvec,
                              tvec)
  else:
    retval = False
    rvec = []
    tvec = []
  return retval, rvec, tvec

#-------------------------------------------------------------------------------
def pose_as_dataclass(array_a:np.array,
                      template:str,
                      calib_a:str,
                      img_a:str)-> RTMatrix:
  """Get RT of camera to board.

  Args:
    array_a (np.array): Numpy array of the image.
    template (str): Template to estimate pose.
    calib_a (str): calibration file for this camera.
    img_a (str): Path to image.

  Raises:
    RegistrationError: Failed to estimate pose of board.

  Returns:
    RTMatrix: Rotation and translation as a dataclass.
  """

  [retval_a, rvec_a, tvec_a] = estimate_pose(array_a, template, calib_a)
  if retval_a is False:
    raise RegistrationError(f"Could not estimate pose for image @ {img_a}")

  pose_a = RTMatrix(rvec_a, tvec_a)
  return pose_a

#-------------------------------------------------------------------------------
def unproject(points:np.array, k_mat:np.array, dist:np.array) -> np.array:
  """
  Take the 2D distorted markers in an image and unproject into normalized 3D
  coordinates.

  Args:
    points (np.array): Location of markers in image.
    k_mat (np.array): Camera Matrix.
    dist (np.array): Distortion coefficients.

  Returns:
    np.array: 3D Normalized coordinates of markers after unprojection.
  """

  principal_point = [k_mat[0][2], k_mat[1][2]]
  focal_length = [k_mat[0][0], k_mat[1][1]]

  term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                   400,
                   0.000001)
  x_u = cv2.undistortPointsIter(points,
                                k_mat,
                                dist,
                                np.eye(3),
                                k_mat,
                                term_criteria)

  rays = (x_u-principal_point)/focal_length
  return rays

#-------------------------------------------------------------------------------
def registration_error(array_a:np.array,
             template:str,
             calib_a:str,
             pose_b:RTMatrix,
             rmat_b_to_a:np.array,
             tvec_b_to_a:np.array) -> Tuple[float, float]:
  """
  Calculate the registration error between two cameras. The registration error
  is computed by taking the rotation and translation of the board to camera B
  coordinates, applying the registration, then projecting into 2D camera A
  coordinates, then calculating this against the reprojection error of the
  detected markers from camera A.

    camera B          camera A (prj)
  ------------        ------------
  | 3D points| -----> | 3D points|
  ------------        ------------
       ^                   |
       |                   |
       |                   v                     camera A (detected)
  ------------        ------------   diff with   ------------
  | 3D board |        | 2D points|  <--------->  | 2D points|
  ------------        ------------               ------------

  Args:
    array_a (np.array): Image from camera A.
    template (str): Template of the board.
    calib_a (str): Path to camera calibration file for camera A.
    pose_b (RTMatrix): Rotation and translation of board to camera B.
    rmat_b_to_a (np.array): Rotation matrix of camera B to camera A.
    tvec_b_to_a (np.array): Translation vector of camera B to camera A.

  Returns:
    Tuple[float, float]: Root mean square of all reprojected points in pixels
      and radians.
  """

  # Get 2d image coordinates of markers detected in camera A.
  corners_a, ids_a, board = detect_markers(array_a, template)

  # 3D board points in coordinates of camera B.
  points_board = board.chessboardCorners[ids_a, :]
  squoze = points_board.squeeze(1)
  markers_cam_b = np.matmul(r_as_matrix(pose_b.rotation),
                            squoze.transpose()) + pose_b.translation

  # Multiply by registration to get markers 3D coordinates in camera A.
  pts_in_cam_a = np.matmul(rmat_b_to_a, markers_cam_b) + tvec_b_to_a

  # Registration computed 3D points to 2D image plane A.
  k_mat, dist, _ = read_opencv_calfile(calib_a)
  pts, _ = cv2.projectPoints(pts_in_cam_a,
                             np.eye(3),
                             np.zeros((3,1)),
                             k_mat,
                             dist)

  # Express difference between measured and projected as radians.
  angles = []
  measured = unproject(corners_a, k_mat, dist)
  prediction = unproject(pts, k_mat, dist)

  for idx, elem in enumerate(measured):
    meas = [elem[0][0], elem[0][1], 1]
    pred = [prediction[idx][0][0], prediction[idx][0][1], 1]

    dot_product = np.dot(meas, pred)
    norm_a = np.linalg.norm(meas)
    norm_b = np.linalg.norm(pred)
    theta = math.acos(dot_product / (norm_a * norm_b))
    angles.append(theta)

  # Get Root Mean Square of measured points in camera A to registration
  # calculated points.
  num_points = pts.shape[0]
  squares = [elem**2 for elem in angles]
  rms_radians = math.sqrt(sum(squares)/num_points)
  print(f"RMS (Radians): {rms_radians}")

  rms_pixels = np.sqrt(np.sum((corners_a-pts)**2)/num_points)
  return rms_pixels, rms_radians

#-------------------------------------------------------------------------------
def calibrate_camera(
  imdir:str,
  template:str,
  init_calfile:str = None,
  rms_thr:float = 1.0,
  postfix:str = "",
  min_detections:int = 30,
  min_images:int = 30,
  min_quality_images:int = 30,
  per_view_threshold:int = 1
  ) -> Tuple[float,
             np.array,
             np.array,
             np.array,
             List,
             List]:
  """  Calibrate a camera using charuco detector and opencv bundler.

  Args:
    imdir (str): Image directory.
    template (str): Fullpath of the board json_file.
    init_calfile (str, optional): Calibration file. Defaults to None.
    rms_thr (float, optional): Reprojection threshold. Defaults to 1.0.
    postfix (str, optional): Calibration filename suffix. Defaults to "".
    min_detections (int, optional): Required minimum number of detections.
      Defaults to 30.
    min_images (int, optional): Minimum number of images. Defaults to 30.
    min_quality_images (int, optional): Minimum number of images with sufficient
      reprojection quality. Defaults to 30.
    per_view_threshold (int, optional): RMS reprojection error threshold to
      distinguish quality images. Defaults to 1.

  Raises:
    CalibrationError: Not enough images for calibration.
    CalibrationError: Not enough detections for calibration.
    CalibrationError: Not enough images with low rms for calibration.

  Returns:
    Tuple[float, np.array, np.array, np.array, List, List]:
      rms: the overall RMS re-projection error in pixels.
      k_matrix: camera matrix.
      dist: Lens distortion coeffs. OpenCV model with 8 distortion is equivalent
        to Brown-Conrady model. (used in K4A).
      img_size: [width; height].
      rvecs: list of rotation vectors.
      tvecs: list of translation vectors.
  """

  # Check validity of initial calibration file.
  if init_calfile is not None and os.path.exists(init_calfile) is False:
    FileExistsError(f"Initial calibration does not exist: {init_calfile}")

  # output cal file
  calfile = os.path.join(imdir, f"calib{postfix}.yml")

  imgnames = []
  for file in os.listdir(imdir):
    if fnmatch.fnmatch(file, "*.png") or fnmatch.fnmatch(file, "*.jpg"):
      imgnames.append(os.path.join(imdir, file))

  num_images = len(imgnames)
  if num_images < min_images:
    msg = f"Not Enough images. {num_images} found, {min_images} required."
    raise CalibrationError(msg)

  ccorners_all, cids_all, p3d, img_size, board = \
    detect_markers_many_images(imgnames, template)

  # check number of times corners were successfully detected.
  num_det = len(ccorners_all)
  if num_det < min_detections:
    msg = f"Insufficent detections. {num_det} found, {min_detections} required."
    raise CalibrationError(msg)

  # initial calibration
  if init_calfile is None:
    # get image size of any image
    k_matrix = cv2.initCameraMatrix2D(p3d, ccorners_all, img_size)  # (w,h)
    dist = np.zeros((8, 1),  dtype=np.float32)
  else:
    k_matrix, dist, img_data = read_opencv_calfile(init_calfile)
    img_arr = img_data.astype(int)
    img_size = (img_arr[0][0], img_arr[1][0])

  flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_USE_INTRINSIC_GUESS
  criteria = cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6

  start = time.perf_counter()
  rms, k_matrix, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = \
        aruco.calibrateCameraCharucoExtended(ccorners_all,
                           cids_all,
                           board,
                           img_size,
                           k_matrix,
                           dist,
                           flags=flags,
                           criteria=criteria)

  # Check Quality of each image.
  n_low_rms = [err[0] for err in perViewErrors if err[0] <= per_view_threshold]
  num_good_images = len(n_low_rms)

  # Report which indexes are failing in perViewErrors.
  failing_idxs = []
  [failing_idxs.append(str(index)) for (index, err) in enumerate(perViewErrors) if err[0] > per_view_threshold]
  warning_failing_indexes = "Failing image indices: " + ", ".join(failing_idxs)

  if len(failing_idxs) != 0:
    warnings.warn(warning_failing_indexes)

  if num_good_images < min_quality_images:
    msg = f"Insufficent number of quality images. " \
      f"{num_good_images} found, {min_quality_images} required."
    raise CalibrationError(msg)

  dist8 = dist[:8, :]
  img_size_as_array = np.array([np.array([img_size[0]]),
                                np.array([img_size[1]])])
  if rms < rms_thr:
    print("calibrate_camera took {} sec".format(time.perf_counter()-start))
    write_opencv_calfile(calfile, k_matrix, dist8, img_size_as_array)
    write_json(os.path.join(imdir, "report.json"), {"RMS_pixels":rms})
  else:
    print("calibrate_camera failed \n")

  return rms, k_matrix, dist, img_size, rvecs, tvecs, num_good_images

#-------------------------------------------------------------------------------
def register(img_a: str,
       img_b: str,
       template: str,
       calib_a: str,
       calib_b: str,
       out_dir: str,
       rms_threshold:float=0.001) -> Tuple[np.array, np.array, float]:
  """Get rotation and translation of camera b in terms of camera a.

  Args:
    img_a (str): Full path to image taken by camera A.
    img_b (str):  Full path to image taken by camera B.
    template (str): Full path to template image of board.
    calib_a (str): Full path to opencv calibration file of camera A.
    calib_b (str): Full path to opencv calibration file of camera B.
    out_dir (str): Output directory for full calibration blob.
    rms_threshold (float): Threshold to fail RMS at in radians.

  Raises:
    FileExistsError: Raise if image file A is not found.
    FileExistsError: Raise if image file B is not found.
    FileExistsError: Raise if template file is not found.
    FileExistsError: Raise if calibration parameters for camera A is not found.
    FileExistsError: Raise if calibration parameters for camera B is not found.
    RegistrationError: Raise if OpenCV fails to load image file A.
    RegistrationError: Raise if OpenCV fails to load image file B.
    RegistrationError: Raise if reprojection error is too large.

  Returns:
    Tuple[np.array, np.array, float]: Return Rotation Translation of camera B
      to A, and rms.
      rmat_b_to_a: Numpy array of the rotation matrix from B to A.
      tmat_b_to_a: Numpy array of the translation vector from B to A.
      rms: Reprojection error expressed as Root mean square of pixel diffs
        between markers.
  """

  # File exists checks.
  if not os.path.exists(img_a):
    raise FileExistsError(f"Image file not found for camera A @ {img_a}")
  if not os.path.exists(img_b):
    raise FileExistsError(f"Image file not found for camera B @ {img_b}")
  if not os.path.exists(template):
    raise FileExistsError(f"Board template parameters not found @ {template}")
  if not os.path.exists(calib_a):
    raise FileExistsError(f"Calib params for camera A not found @ {calib_a}")
  if not os.path.exists(calib_b):
    raise FileExistsError(f"Calib params for camera B not found @ {calib_b}")

  array_a = cv2.imread(img_a)
  array_b = cv2.imread(img_b)

  # Check image was read by opencv.
  if array_a is None:
    raise RegistrationError(f"OpenCV could not interpret Camera A @ {img_a}")
  if array_b is None:
    raise RegistrationError(f"OpenCV could not interpret Camera B @ {img_b}")

  # Get Rt of camera A to board.
  pose_a = pose_as_dataclass(array_a, template, calib_a, img_a)
  rmat_a = r_as_matrix(pose_a.rotation)

  # Get Rt of camera B to board.
  pose_b = pose_as_dataclass(array_b, template, calib_b, img_b)
  rmat_b = r_as_matrix(pose_b.rotation)

  # Get perspective of camera B to board.
  rmat_b_to_a = np.matmul(rmat_a, rmat_b.transpose())
  tvec_b_to_a = -np.matmul(rmat_b_to_a, pose_b.translation) + pose_a.translation

  print(f"Translation camera B to A:\n{tvec_b_to_a}")
  print(f"Rotation camera B to A:\n{rmat_b_to_a}")

  # Find registration error.
  print("Find forward reprojection error (camera B to camera A).")
  (rms1_pixels, rms1_rad) = registration_error(array_a,
                         template,
                         calib_a,
                         pose_b,
                         rmat_b_to_a,
                         tvec_b_to_a)
  if rms1_rad > rms_threshold:
    raise RegistrationError("Registration error from A to B too large.")

  # Find reverse registration error.
  print("Find reverse reprojection error (camera A to camera B).")
  (rms2_pixels, rms2_rad) = registration_error(array_b,
                         template,
                         calib_b,
                         pose_a,
                         rmat_b_to_a.transpose(),
                         rmat_b_to_a.transpose() @ tvec_b_to_a*-1)
  if rms2_rad > rms_threshold:
    raise RegistrationError("Registration error from B to A too large.")

  # Write to calibration blob.
  write_calibration_blob([calib_a, calib_b], rmat_b_to_a, tvec_b_to_a, out_dir)
  rms_report = {"RMS_B_to_A_pixels": rms1_pixels,
          "RMS_B_to_A_radians": rms1_rad,
          "RMS_A_to_B_pixels": rms2_pixels,
          "RMS_A_to_B_radians": rms2_rad}
  write_json(os.path.join(out_dir, "report.json"), rms_report)

  return rmat_b_to_a, tvec_b_to_a, rms1_pixels, rms1_rad, rms2_pixels, rms2_rad
