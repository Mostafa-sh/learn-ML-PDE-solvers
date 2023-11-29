
"""
Developed by Mostafa Shojaei
2023
"""

# ===========================================================
# TO DO 0: 
# Read and understand the following code 
# and complete all TO DO boxes. 
# ===========================================================

import numpy as np

def get_coordinate_rec_mesh(mesh_size):
    """
    This function takes mesh_size = (nx, ny) and generates 
    the coordinates of nodes in a structured rectangular mesh
    for a unit square domain
    """
    nx,ny = mesh_size
    x,y = np.meshgrid(np.linspace(0, 1, nx),np.linspace(0, 1, ny))
    # ===========================================================
    # TO DO 1: 
    # Make a 2D numpy array of size (2 x nx*ny) containing 
    # x and y coordinates. This can be done in one line.
    output = None
    # ===========================================================
    return output


def get_connection_rec_mesh(mesh_size):
    """
    This function takes mesh_size = (nx, ny) and  generates 
    the connectivity table ct for a structured rectangular mesh
    using anticlockwise numbering
    """
    nx, ny  = mesh_size
    local = np.array([0, 1, nx+1, nx])
    ct = np.array([local + i+j*nx for j in range(ny-1) for i in range(nx-1)]).T
    # ===========================================================
    # TO DO 2: 
    # Here ct is a 2D numpy array of size 4 x number of elements.
    # Each column contains node numberings of each element.
    # Looking at this function, what is the number of elements? 
    # EXPLAIN your answer here as comments.
    #
    # ===========================================================
    return ct


def local_shape_functions(x,y):
    """
    This function takes coordinates x and y and returns 
    shape function values and their derivatives in the 
    reference element.
    """
    # ===========================================================
    # TO DO 3: 
    # Write shape functions and their corresponding derivatives.
    # Must hold: b1(-1,-1) = b2(1,-1) = b3(1,1) = b4(-1,1) = 1.
    b1 = None
    b2 = None
    b3 = None
    b4 = None
    ## ----------
    bx1 = None
    bx2 = None
    bx3 = None
    bx4 = None
    ## ----------
    by1 = None
    by2 = None
    by3 = None
    by4 = None
    # ===========================================================
    b  = [b1, b2, b3, b4]
    bx = [bx1, bx2, bx3, bx4]
    by = [by1, by2, by3, by4]
    b = np.array(b)
    Gb = np.array([bx,by])  # Gb is gradient of b  
    return b, Gb

def gauss_int_points(ng=2):
    """
    This function takes number of Gaussian points ng
    in each direction and returns Gaussian weights and points
    """
    if ng == 1:
        xg = [0]
        wg = [2]
    elif ng == 2:
        # ===========================================================
        # TO DO 4: 
        # Write Gaussian weights and points for ng = 2.
        xg = []
        wg = []
        # ===========================================================
    elif ng == 3:
        xg =[-np.sqrt(3/5), 0, +np.sqrt(3/5)]
        wg = [5/9, 8/9, 5/9]
    x,y = np.meshgrid(xg,xg)
    w = np.array([wg]).T*wg
    return x.flatten(), y.flatten(), w.flatten()


def get_gauss_int(fun,ng):
    """
    This function takes a function fun(x,y) and 
    number of Gaussian points ng and computes and returns 
    the Gaussian integral of the function
    """
    xg,yg,wg= gauss_int_points(ng)
    intg = 0
    # ===========================================================
    # TO DO 5: 
    # Write ONE for loop to compute Gaussian integral
    # using Gauss points xg, yg and Gauss weights wg.


    # ===========================================================
    return intg


def get_element_jacobian(xe,ye):
    """
    This function takes the coordinates of 4 nodes (xe, ye) of any 
    elements in the mesh and returns the Jacobian transformation 
    matrix of that element.
    """
    # ===========================================================
    # TO DO 6: 
    # EXPLAIN why the following gives the Jacobin 
    # transformation. What assumptions result in a constant 
    # Jacobian matrix for all elements in the mesh?
    #
    # ===========================================================
    return np.array([[(max(xe)-min(xe))/2,0],[0.0,(max(ye)-min(ye))/2]])


def assemble_constant_mat(M,ct):
    """
    This function takes a matrix M and connectivity table ct
    and returns an assembled global matrix 
    """
    n_local_dofs , n_elements = np.shape(ct)
    a,b = np.shape(M)
    assert(a==b)
    assert(n_local_dofs==a)
    total_dofs = np.max(np.max(ct))+1

    assembled_mat = np.zeros((total_dofs,total_dofs))
    # ===========================================================
    # TO DO 7: 
    # (a) EXPLAIN why the local matrix of each element
    # is the same for all elements. Hint: there are 2 reasons.
    # (b) Write ONE for loop to assemble the input local matrix M
    # for all elements in the mesh using the connectivity table ct.


    # ===========================================================
    return assembled_mat


def assemble_constant_vec(V,ct):
    """
    This function takes a vector V and connectivity table ct
    and returns an assembled global vector 
    """
    n_local_dofs , n_elements = np.shape(ct)
    a = np.shape(V)[0]
    assert(n_local_dofs==a)
    total_dofs = np.max(np.max(ct))+1
    
    assembled_vec = np.zeros(total_dofs)
    # ===========================================================
    # TO DO 8: 
    # Write ONE for loop to assemble the input local vector V
    # for all elements in the mesh using the connectivity table ct.


    # ===========================================================
    return assembled_vec


"""
--------------------------------------------
We now use the above general FEM functions
to solve a linear PDE in 2D.
This example solves:
div(grad(u)) + 1 = 0
--------------------------------------------
"""

def get_K_and_F(nx,ny,ng=2):
    """
    This function takes mesh size nx x ny 
    and the number of Gauss points ng and 
    returns the global K and F for solving
    div(grad(u)) + 1 = 0 such that KU = F
    """
    mesh_size  = (nx,ny)
    cord = get_coordinate_rec_mesh(mesh_size)
    ct = get_connection_rec_mesh(mesh_size)
    # ===========================================================
    # TO DO 9: 
    # EXPLAIN what happens in the following 2 lines.
    #
    # ===========================================================
    xx = cord[0][ct]
    yy = cord[1][ct]

    J = get_element_jacobian(xx[:,0],yy[:,0])
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)

    # ===========================================================
    # TO DO 10: 
    # Defie a function here called local_K that takes x,y as
    # inputs and returns the local stiffness matrix of elements.
    # Hint: the output should be a 4 x 4 numpy array.
    # In Python, matrix multiplication is done by @, e.g., A @ B.
    # Also see TO DO 11.




    # ===========================================================

    # ===========================================================
    # TO DO 11: 
    # EXPLAIN here the purpose of local_F(x,y) function 
    # and how it works.
    #
    # ===========================================================
    def local_F(x,y):
        b =  local_shape_functions(x,y)[0]
        return 1 * b * detJ
    
    # ===========================================================
    # TO DO 12: 
    # Write the entries/inputs of the following two functions.
    Ke = get_gauss_int()
    Fe = get_gauss_int()
    # ===========================================================

    # ===========================================================
    # TO DO 13: 
    # Use assemble_constant_mat() and assemble_constant_vec() 
    # defined above to assemble Ke and Fe for all elements.
    # Assembled Ke and Fe should be called K and F.


    # ===========================================================
    return K, F


def solve(img,ng=2):
    """
    This function takes a matrix img containing 
    the boundary info and returns the solution of 
    div(grad(u)) + 1 = 0 for the given boundary 
    conditions.
    """
    ny,nx = img.shape

    K,F = get_K_and_F(nx,ny,ng)

    #boundary
    dom_ind = (img == -2).flatten()
    bc_ind1 = np.logical_and(np.logical_not(dom_ind),(img != 0).flatten())
 
    U1 = img.flatten()[bc_ind1]
    F1 = K[:,bc_ind1] @ U1

    # ===========================================================
    # TO DO 14: 
    # Use dom_ind to remove rows and columns of K 
    # corresponding to Dirichlet boundary conditions and
    # call the resulting reduced matrix Kd.
    # Hint: see Fd.


    # ===========================================================

    # ===========================================================
    # TO DO 15: 
    # EXPLAIN what is F1.
    #
    # ===========================================================
    Fd = F[dom_ind] - F1[dom_ind]

    # solve
    U = np.zeros(nx*ny)
    U[bc_ind1] = U1
    U[dom_ind] = np.linalg.solve(Kd, Fd)

    return U