import numpy as np
import dlib
import cv2
from sklearn.utils import shuffle
from math import *

def generate_doublets(images_dataset, labels_dataset):
  unique_labels = np.unique(labels_dataset)
  label_wise_indices = dict()
  for label in unique_labels:
      label_wise_indices.setdefault(label, [index for index, curr_label in enumerate(labels_dataset) if label == curr_label] )
  
  doublets_images = []
  doublets_labels = []
  for index, image in enumerate(images_dataset):
    pos_indices = label_wise_indices.get(labels_dataset[index])
    pos_image = images_dataset[np.random.choice(pos_indices)]
    doublets_images.append((image, pos_image))
    doublets_labels.append(1)

    neg_indices = np.where(labels_dataset != labels_dataset[index])
    neg_image = images_dataset[np.random.choice(neg_indices[0])]
    doublets_images.append((image, neg_image))
    doublets_labels.append(0)

  return shuffle(np.array(doublets_images), np.array(doublets_labels), random_state=0)

def generate_triplets(images_dataset, labels_dataset):
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(label, [index for index, curr_label in enumerate(labels_dataset) if label == curr_label] )
    
    triplets_images = []
    triplets_labels = []
    for index, image in enumerate(images_dataset):
      pos_indices = label_wise_indices.get(labels_dataset[index])
      pos_image = images_dataset[np.random.choice(pos_indices)]

      neg_indices = np.where(labels_dataset != labels_dataset[index])
      neg_image = images_dataset[np.random.choice(neg_indices[0])]
      triplets_images.append((image, pos_image, neg_image))

    return np.array(triplets_images)

def make_patchs(dataset_images, labels, x_size=94, y_size=125):
  """
  Fonction qui renvoie un array avec patch[0] les visages, patch[1] les deux yeux, patch[2] les bouches, patch[3] les nez, patch[4]    les machoirs, patch[5] les yeux gauches et patch[6] les yeux droits 
  """

  dataset_images, labels = shuffle(dataset_images, labels, random_state=0)

  len_train, len_test= int(0.75*len(dataset_images)), int(0.2*len(dataset_images))

  y_train = labels[:len_train]
  y_test = labels[len_train:len_train+len_test]
  y_validation = labels[len_train+len_test:]

  patchs_train = [np.empty(0) for i in range(7)]
  patchs_test = [np.empty(0) for i in range(7)]
  patchs_validation = [np.empty(0) for i in range(7)]

  hauteurs = np.array([0 for i in range(7)])
  largeurs = np.array([0 for i in range(7)])
  centres = np.zeros((7, len(dataset_images), 2))

  bouches = []
  yeux_gauches = []
  yeux_droits = []
  yeux = []
  nezs = []
  machoires = []

  detector = dlib.get_frontal_face_detector()
  # Load the predictor
  predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

  for i in range(len(dataset_images)) :
    # Fait la detection
    img = dataset_images[i,:,:,:].astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img, 1)
    for face in faces:
      landmarks = predictor(image=gray, box=face)

    # Prend les repères
    points = np.array([np.array((landmarks.part(i).x, landmarks.part(i).y)) for i in range(0, 68)])
    yeux.append(np.concatenate((points[22:27], points[42:48], points[17:22], points[36:42])))
    bouches.append(points[48:68])
    nezs.append(points[27:36])
    machoires.append(points[3:13])
    yeux_gauches.append(np.concatenate((points[22:27], points[42:48])))
    yeux_droits.append(np.concatenate((points[17:22], points[36:42])))
  
  yeux = np.array(yeux)
  bouches = np.array(bouches)
  nezs = np.array(nezs)
  machoires = np.array(machoires)
  yeux_gauches = np.array(yeux_gauches)
  yeux_droits = np.array(yeux_droits)

  points_landmarks = np.array([np.array([(0,0)]), yeux, bouches, nezs, machoires, yeux_gauches, yeux_droits], dtype='object')

  # Mets sous la forme de array pour calculer le max et min plus facilement
  for i in range(7):
    points_landmarks[i] = np.array(points_landmarks[i])

  # Défini la hauteur la largeur et le centre pour le dataset
  for i in range(1, 7):
    hauteurs[i] = min(np.max(np.array([np.max(partie[:,1]) - np.min(partie[:,1]) for partie in points_landmarks[i][:len_train]])), y_size)
    largeurs[i] = min(np.max(np.array([np.max(partie[:,0]) - np.min(partie[:,0]) for partie in points_landmarks[i][:len_train]])), x_size)
    centres[i] = np.array([((np.max(partie[:,0]) + np.min(partie[:,0]))//2, (np.max(partie[:,1]) + np.min(partie[:,1]))//2) for partie in points_landmarks[i]])
  
  # Ajout des images entières
  patchs_train[0] = dataset_images[:len_train]
  patchs_test[0] = dataset_images[len_train:len_train+len_test]
  patchs_validation[0] = dataset_images[len_train+len_test:]
  
  # Ajout des autres patch pour le train
  for i in range(1, 7):
    patch = []
    for j in range(len(dataset_images)):
      if ceil(centres[i][j][0] + largeurs[i]/2) > x_size :
        if ceil(centres[i][j][1] + hauteurs[i]/2) > y_size :
          patch.append(dataset_images[j, y_size - hauteurs[i]:, x_size - largeurs[i]:,:])
        else :
          if ceil(centres[i][j][1] - hauteurs[i]/2) < 0 :
            patch.append(dataset_images[j,:hauteurs[i], x_size - largeurs[i]:,:])
          else :
            patch.append(dataset_images[j, ceil(centres[i][j][1] - hauteurs[i]/2):ceil(centres[i][j][1] + hauteurs[i]/2), x_size - largeurs[i]:,:])
      else :
        if ceil(centres[i][j][1] + hauteurs[i]/2) > y_size :
          if ceil(centres[i][j][0] - largeurs[i]/2) < 0 :
            patch.append(dataset_images[j, y_size - hauteurs[i]:, :largeurs[i],:])
          else :
            patch.append(dataset_images[j, y_size - hauteurs[i]:, ceil(centres[i][j][0] - largeurs[i]/2):ceil(centres[i][j][0] + largeurs[i]/2),:])
        else :
          if ceil(centres[i][j][0] - largeurs[i]/2) < 0 :
            if ceil(centres[i][j][1] - hauteurs[i]/2) < 0 :
              patch.append(dataset_images[j, :hauteurs[i], :largeurs[i],:])
            else :
              patch.append(dataset_images[j, ceil(centres[i][j][1] - hauteurs[i]/2):ceil(centres[i][j][1] + hauteurs[i]/2), :largeurs[i],:])
          else :
            if ceil(centres[i][j][1] - hauteurs[i]/2) < 0 :
              patch.append(dataset_images[j, :hauteurs[i], ceil(centres[i][j][0] - largeurs[i]/2):ceil(centres[i][j][0] + largeurs[i]/2),:])
            else :
              patch.append(dataset_images[j, ceil(centres[i][j][1] - hauteurs[i]/2):ceil(centres[i][j][1] + hauteurs[i]/2),  ceil(centres[i][j][0] - largeurs[i]/2):ceil(centres[i][j][0] + largeurs[i]/2),:])

    # Ajout des images
    patchs_train[i] = np.array(patch[:len_train])
    patchs_test[i] = np.array(patch[len_train:len_train+len_test])
    patchs_validation[i] = np.array(patch[len_train+len_test:])
  
  return patchs_train, patchs_test, patchs_validation, y_train, y_test, y_validation


def rearrange_arr_triplet(arr):
  """
  Réarange les images de (hauteur, largeur, RGB) => (RGB, hauteur, largeur)
  """
  dim_x, dim_y = arr.shape[3], arr.shape[2]
  new_array1 = np.zeros((len(arr), 3, 3, dim_x, dim_y))
  new_array2 = np.zeros((len(arr), 3, 3, dim_y, dim_x))
  new_array3 = [[] for _ in range(len(arr))]
  for i in range(len(arr)):
    for j in range(3):
      new_array1[i,j] = arr[i,j,:,:,:].T
  for i in range(len(arr)):
    for j in range(3):
      for y in range(3):
        new_array2[i,j,y] = new_array1[i,j,y,:,:].T
  for i in range(len(arr)):
    new_array3[i] = tuple(new_array2[i])
  return new_array3

def rearrange_arr_doublet(arr):
  dim_x, dim_y = arr.shape[3], arr.shape[2]
  new_array1 = np.zeros((len(arr), 2, 3, dim_x, dim_y))
  new_array2 = np.zeros((len(arr), 2, 3, dim_y, dim_x))
  new_array3 = [[] for _ in range(len(arr))]
  for i in range(len(arr)):
    for j in range(2):
      new_array1[i,j] = arr[i,j,:,:,:].T
  for i in range(len(arr)):
    for j in range(2):
      for y in range(3):
        new_array2[i,j,y] = new_array1[i,j,y,:,:].T
  for i in range(len(arr)):
    new_array3[i] = tuple(new_array2[i])
  return new_array3


def dataset_config_triplet(patchs_train, patchs_validation, y_train, y_validation):
    """
    In charge of all the necessary configurations of the Dataset
    """
        
    patchs_train = [patchs_train[i] / 255. for i in range(len(patchs_train))]
    patchs_validation = [patchs_validation[i] / 255. for i in range(len(patchs_validation))]

    patch_triplets_train = [generate_triplets(patchs_train[i], np.argmax(y_train, axis=1)) for i in range(7)]
    # patch_triplets_test = [generate_triplets(patchs_test[i], np.argmax(y_test, axis=1)) for i in range(7)]
    patch_triplets_validation = [generate_triplets(patchs_validation[i], np.argmax(y_validation, axis=1)) for i in range(7)]

    data_tr = [rearrange_arr_triplet(triplet_train_dataset) for triplet_train_dataset in patch_triplets_train]
    data_val = [rearrange_arr_triplet(triplet_validation_dataset) for triplet_validation_dataset in patch_triplets_validation]

    Data_train = [[] for _ in range(7)]
    Data_val = [[] for _ in range(7)]
    for i in range(7):
        for index, target in enumerate(y_train) :
            Data_train[i].append((data_tr[i][index], target))

    for index, target in enumerate(y_validation) :
        Data_val[i].append((data_val[i][index], target))    
    
    return Data_train, Data_val


def dataset_config_doublet(patchs_train, patchs_validation, y_train, y_validation):
    """
    In charge of all the necessary configurations of the Dataset
    """
        
    patchs_train = [patchs_train[i] / 255. for i in range(len(patchs_train))]
    patchs_validation = [patchs_validation[i] / 255. for i in range(len(patchs_validation))]

    patch_doublets_train = [generate_doublets(patchs_train[i], np.argmax(y_train, axis=1)) for i in range(7)]
    # patch_doublets_test = [generate_doublets(patchs_test[i], np.argmax(y_test, axis=1)) for i in range(7)]
    patch_doublets_validation = [generate_doublets(patchs_validation[i], np.argmax(y_validation, axis=1)) for i in range(7)]

    data_tr = [rearrange_arr_doublet(doublets_train_dataset) for doublets_train_dataset in patch_doublets_train]
    data_val = [rearrange_arr_doublet(doublets_validation_dataset) for doublets_validation_dataset in patch_doublets_validation]

    Data_train = [[] for _ in range(7)]
    Data_val = [[] for _ in range(7)]
    for i in range(7):
        for index, target in enumerate(y_train) :
            Data_train[i].append((data_tr[i][index], target))

    for index, target in enumerate(y_validation) :
        Data_val[i].append((data_val[i][index], target))    
    
    return Data_train, Data_val

