"""
Original file is located at
  https://colab.research.google.com/drive/1Q5fpOqGYIc2kVXbhptZchQ0H2R4DCJzO?usp=sharing

Github repository url at
  https://github.com/GG-kun/letter_detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt #graficar el resultado
import math

def slope(a, b):
  x0, y0, z = a.ravel()
  x1, y1, z = b.ravel()

  y = y1 - y0
  if abs(y) <= 5:
    y = 0
  
  x = x1 - x0
  if abs(x) <= 5:
    return 0
  return y/x

def distance(a, b):
  x0, y0, z = a.ravel()
  x1, y1, z = b.ravel()

  y = y1 - y0
  x = x1 - x0

  return math.sqrt(y*y + x*x)

def get_pair_index(point, matrix, previous_point):

    min_distance = float('inf')
    pair_index = None
    for j, pair in enumerate(matrix):
      if slope(point, pair) == 0:
        current_distance = distance(point, pair)
        if distance(point, previous_point) == 0 or slope(previous_point, pair) != 0:
          if current_distance < min_distance:
            min_distance = current_distance
            pair_index = j
    return pair_index

def matrix_remove_z(matrix):
  matrix = matrix.copy()
  matrix = np.delete(matrix, -1, axis=1)

  return matrix

def set_points(image, points, color=255):
  x, y, channels = image.shape

  circle_radius = int(x * 0.005)
  for point in points:
      x,y,z = point.ravel()
      cv2.circle(image,(x,y),circle_radius,color,-1)

  return image

def set_poly(image, points, color=(255, 0, 0)):

  points = matrix_remove_z(points)
  points = points.astype('int32')
  cv2.fillPoly(image, [points], color)

  return image

# Only works for 3x3 matrix
def inverse_matrix(matrix):
  a, b, x = matrix[0].ravel()
  c, d, y = matrix[1].ravel()
  m, n, z = matrix[2].ravel()

  det = 1 / (a*d - b*c)
  det_matrix = np.array([
                         [d, -b, 0],
                         [-c, a, 0],
                         [c*n - d*m, b*m - a*n, a*d-c*b],
  ])

  return det * det_matrix

def solve_system(coefficient_matrix, constant_matrix):

  t = np.linalg.inv(coefficient_matrix).dot(constant_matrix)

  return np.array([
                   [t[0], t[1], 0],
                   [t[2], t[3], 0],
                   [t[4], t[5], 1],
  ])

# Retrieves a given point [x, y, z] in a variable matrix [a, b, c, d, m, n]
def point_to_variables(point):

  x, y, z = point.ravel()

  return np.array([
                   [x, 0, y, 0, z, 0],
                   [0, x, 0, y, 0, z],
  ])

def get_coefficient_matrix(points):

  coefficient_matrix = point_to_variables(points[0])
  for point in points[1:]:
    coefficient_matrix = np.concatenate((coefficient_matrix, point_to_variables(point)))

  return coefficient_matrix

def get_constant_matrix(points):

  constant_matrix = []
  for point in points:
    x, y, z = point.ravel()
    constant_matrix.append(x)
    constant_matrix.append(y)

  return np.array(constant_matrix)

def reconstruct_transformation(begin_points, end_points):

  coefficient_matrix = get_coefficient_matrix(begin_points)
  print(coefficient_matrix)

  constant_matrix = get_constant_matrix(end_points)
  print(constant_matrix)

  t = solve_system(coefficient_matrix, constant_matrix)

  return t

def translate_matrix(x, y, matrix):
  affine_matrix = np.array([
                            [1, 0, 0],
                            [0, 1, 0],
                            [x, y, 1],
  ])
  return np.matmul(matrix.copy(), affine_matrix)

def rotate_matrix(theta, matrix):
  sign = math.copysign(1, theta)
  theta = math.radians(theta)

  cost = math.cos(theta)
  sint = math.sin(theta)
  
  affine_matrix = np.array([
                            [cost, -1*sign*sint, 0],
                            [sign*sint, cost, 0],
                            [0, 0, 1],
  ])
  return np.matmul(matrix.copy(), affine_matrix)

def scale_matrix(x, y, matrix):
  affine_matrix = np.array([
                            [x, 0, 0],
                            [0, y, 0],
                            [0, 0, 1],
  ])
  return np.matmul(matrix.copy(), affine_matrix)

def sort_matrix(matrix):
  sorted_matrix = []
  missing_matrix = np.copy(matrix)

  point = matrix[0]

  pair_point = None
  pair_index = 0
  previous_point = matrix[0]

  while point is not None:
    sorted_matrix.append(point)
    matrix = np.delete(matrix, pair_index, axis=0)

    pair_index = get_pair_index(point, matrix, previous_point)
    if pair_index is None:
      print("Could not find a pair for {0}".format(point))
      break
    else:
      pair_point = matrix[pair_index]
      print("{0},{1}".format(point, pair_point))
    previous_point = point.copy()
    point = pair_point.copy()

  return np.array(sorted_matrix)

def image_matrix(img, maxCorners):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  gray = np.float32(gray)
  corners = cv2.goodFeaturesToTrack(gray,maxCorners,0.01,10)
  corners = np.int0(corners)

  points = []
  for i in corners:
      x,y = i.ravel()
      points.append([x, y, 1])

  return np.array(points)

def image_sorted_matrix(img, minPoints=8, maxCorners=12):
  sorted_matrix = []

  while len(sorted_matrix) < minPoints and maxCorners >= minPoints:
    matrix = image_matrix(img, maxCorners)
    maxCorners -= 1
    sorted_matrix = sort_matrix(matrix)

  return sorted_matrix

def int_matrix(matrix):
  return np.rint(matrix.copy()).astype('int32')

def get_template_matrix(matrix):
  return int_matrix(scale_matrix(factorSize, factorSize,matrix.copy()))

def load_template(file_name, size=None):
  img = cv2.imread(file_name)

  if size is not None:
    img = cv2.resize(img, (size,size))

  template_matrix = image_sorted_matrix(img)

  template_matrix = get_template_matrix(template_matrix)

  # Show
  # img = set_descriptor(img, template_matrix)
  # plt.figure(figsize=(30,15))
  # plt.imshow(img)

  return template_matrix

def return_template(matrix):

  for i, template in enumerate(templates):
    if len(template) == len(matrix):
      return (i, template)
  
  return None

def detect_letter(file_name):
  img = cv2.imread(file_name)

  sorted_matrix = image_sorted_matrix(img)

  template_index, template_matrix = return_template(sorted_matrix)

  if template_matrix is None:
    print("Could not find letter")
  else:
    image = np.zeros([size,size,3], dtype=np.uint8)
    
    t = reconstruct_transformation(template_matrix[:3], sorted_matrix[:3])

    image = set_poly(image, sorted_matrix, color=(0, 255, 0))

    letter_color = 255 / (len(templates)-1)
    for i, template in enumerate(templates):
      if i != template_index:
        template_transformed = np.matmul(template, t)
        image = set_poly(image, template_transformed, color=(letter_color*i, 0, 0))

    # plt.figure(figsize=(30,15))
    # plt.imshow(image)
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.show()

def set_descriptor(image, matrix):
  image = set_poly(image, matrix, color=(0, 0, 255))
  first_point = np.array(matrix[:1])
  image = set_points(image, first_point, color=(0, 255, 0))
  last_point = np.array(matrix[-1:])
  image = set_points(image, last_point, color=(255, 0, 0))

  return image

def debugg_letter(file_name):
  img = cv2.imread(file_name)

  sorted_matrix = image_sorted_matrix(img)

  template_index, template_matrix = return_template(sorted_matrix)

  if template_matrix is None:
    print("Could not find letter")
  else:
    print(template_matrix)

  image = np.zeros([size,size,3], dtype=np.uint8)

  # Show
  image = set_descriptor(image, template_matrix)
  image = set_descriptor(image, sorted_matrix)

  plt.figure(figsize=(30,15))
  plt.imshow(image)

# Global variables
size = 1000
desiredSize = 50
factorSize = desiredSize/size
templates = []

# F template
templates.append(load_template('f/original.jpg'))

# E template
templates.append(load_template('e/original.jpg'))

# I template
templates.append(load_template('i/original.png', size=size))

# T template
templates.append(load_template('t/original.png'))


if __name__ == "__main__":
  
  print(len(templates))

#   detect_letter('t/rotated_90.png')

#   detect_letter('t/rotated_180.png')

#   detect_letter('t/rotated_270.png')

#   detect_letter('f/rotated_270.jpg')

#   # debugg_letter('e/rotated_90.jpg')
  detect_letter('e/rotated_90.jpg')