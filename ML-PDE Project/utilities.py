"""
Developed by Mostafa Shojaei
2023
"""

"""
Do NOT modify this file
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# def plot_mesh_as_image(img):
  
def plot_u(U,img):
    m,n = U.shape
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)
    xp,yp = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,m))
    up = np.reshape(U,(m,n))
    #boundary
    bc_ind = (img != -2)
    up[bc_ind] = img[bc_ind]
    min_u = np.round(np.min(U),4)
    max_u = np.round(np.max(U),4)
    cp = ax.contourf(xp, yp, up)
    plt.title('u(x,y), range = [' + str(min_u) + ', ' + str(max_u) + ']' )
    ax.set_aspect('equal', adjustable='box')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(cp, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    plt.show()

def plot_results(img,U):
    """
    (1) plot mesh as an RGB image
    """
    d0 = img == 0
    d1 = np.logical_and(img != -2, img !=0)  
    ny,nx = img.shape
    R = np.zeros((ny,nx))
    G = np.zeros((ny,nx))
    R[d0] = 1
    G[d1] = 1
    img = np.zeros((ny,nx,3))
    img[:,:,0] = R
    img[:,:,1] = G
    img = img[::-1,:,:]
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title("Input image")
    """
    (2) plot U
    """
    xp,yp = np.meshgrid(np.linspace(0,1,nx),np.linspace(0,1,ny))
    up = np.reshape(U,(ny,nx))    
    ax = fig.add_subplot(1,2,2)
    min_u = np.round(np.min(U),4)
    max_u = np.round(np.max(U),4)
    cp = ax.contourf(xp, yp, up)
    plt.title('u(x,y), range = [' + str(min_u) + ', ' + str(max_u) + ']' )
    ax.set_aspect('equal', adjustable='box')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(cp, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    plt.show()


def make_inpt_image(m,bc_value):
  """
  This function takes boundary values "bc_value = [bottom, top, left, right]"
  of a square image of m x m and return the corresponding image matrix 
  such that all domain pixels have value of -2.
  """
  img = np.zeros((m,m)) - 2
  if bc_value[0] is not None: img[0, :] = bc_value[0]  #bottom
  if bc_value[1] is not None: img[-1,:] = bc_value[1]  #top
  if bc_value[2] is not None: img[:, 0] = bc_value[2]  #left
  if bc_value[3] is not None: img[:,-1] = bc_value[3]  #right
  return img


def add_channel_to_one(img):
    return np.reshape(img,(img.shape[0],img.shape[0],1))


def repeat_one_image(img,k=1):
    """
    repeat a matrix k times along axis 0
    use for data augmentation
    """
    return np.repeat([img],k,axis=0)


class LayerFillRandomNumber(tf.keras.layers.Layer):
    """ 
    A customized Keras layer to insert uniform random values in (0,1) 
    in image pixels if pixels value == -2.0
    """
    def __init__(self, name='fill-random-num'):
        super(LayerFillRandomNumber, self).__init__(name=name)

    def call(self, input):
        output = input + tf.where(
            input > -1.5, tf.fill(tf.shape(input), 0.0),
            tf.random.uniform(tf.shape(input), minval=0.0, maxval=1.0)) + tf.where(
                input > -1.5, tf.fill(tf.shape(input), 0.0),
                tf.fill(tf.shape(input), 2.0))
        return tf.cast(output,dtype=tf.double)

