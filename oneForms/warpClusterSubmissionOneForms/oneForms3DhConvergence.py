# we load the things!

from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.meshes import MakeStructured3DMesh
from ngsolve.solvers import GMRes
from netgen.occ import *
from scipy.optimize import curve_fit
import scipy.sparse as sp
import numpy as np
import pandas as pd

# some helper functions
def logspace_custom_decades(start, stop, points_per_decade):
    
    result = []
    current_decade = start
    while current_decade < stop:
        decade_points = np.logspace(np.log10(current_decade), np.log10(current_decade * 10), points_per_decade, endpoint=False)
        result.extend(decade_points)
        current_decade *= 10
    return np.array(result)

# functions for differential operators on manufactured solutions 

coords = [x,y,z]

def JacobianOfCF(cf):
    """ Function to compute the Jacobi Matrix of a vector coefficient function cf """

    Jac_u_3D = CF((
    cf[0].Diff(x), cf[0].Diff(y), cf[0].Diff(z),
    cf[1].Diff(x), cf[1].Diff(y), cf[1].Diff(z),
    cf[2].Diff(x), cf[2].Diff(y), cf[2].Diff(z)
    ), dims=(3, 3))

    return Jac_u_3D

def GGrad(cf):
    """ Function to compute the gradient of a scalar Coefficient Function """
    gg = [cf.Diff(coords[i]) for i in range(mesh.dim)]
    return CF(tuple(gg))


def GCurl(cf):
    """ Function to compute the curl or rot of vec cf using Jacobian """

    if cf.dim == 1: # if the functions is getting handed a scalar field, its to calculate the curl of the rot..
        curl_rot_u = CF((cf.Diff(y), - cf.Diff(x)))
        return curl_rot_u

    elif mesh.dim == 2:
        rot_u = CF(cf[1].Diff(x) - cf[0].Diff(y))
        return rot_u
    
    elif mesh.dim == 3:
        Jac_u = JacobianOfCF(cf)
        curl_u = CF((Jac_u[2,1] - Jac_u[1,2],  
                    Jac_u[0,2] - Jac_u[2,0],  
                    Jac_u[1,0] - Jac_u[0,1]))
        return curl_u
    

def GDiv(cf):
    """ Function to compute the divergence of a vector coefficient function """

    gd = [cf[i].Diff(coords[i]) for i in range(cf.dim)]
    return CF(sum(gd))

# Functions for plotting, linear regression fit line for convergence

def reference_line_func(h_values, scaling_factor, slope):

    return scaling_factor * h_values ** slope

def fit_reference_line(h_values, error_values):

    popt, _ = curve_fit(reference_line_func, h_values, error_values, p0=[1, 1])

    scaling_factor, slope = popt
    return scaling_factor, slope

# Functions to calculate h_max

def edge_length(v1, v2, dim):
    return np.sqrt(sum((v1[i] - v2[i])**2 for i in range(dim)))

def squared_distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sum((v1 - v2) ** 2)

def cayley_menger_matrix(vertices):
    if len(vertices) != 4:
        raise ValueError("This method is for a tetrahedron, which requires exactly 4 vertices.")

    # Create the Cayley-Menger matrix (5x5)
    C = np.ones((5, 5))
    for i in range(5):
        C[i, i] = 0 

    for i in range(1, 5):
        for j in range(i+1, 5):
            C[i, j] = C[j, i] = squared_distance(vertices[i-1], vertices[j-1])

    return C

def triangle_area(a, b, c):
    s = (a + b + c) / 2  
    return np.sqrt(s * (s - a) * (s - b) * (s - c))

def circumradius_2D(a, b, c):
    area = triangle_area(a, b, c)
    return a * b * c / (4 * area)

def circumradius_3D(vertices):
    C = cayley_menger_matrix(vertices)

    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        raise ValueError("Cayley-Menger matrix is singular or not invertible.")

    M = -2 * C_inv
    circumradius = 0.5 * np.sqrt(M[0, 0])

    return circumradius

def calc_hmax(mesh):
    max_h = 0 
    if mesh.dim == 2:
        for el in mesh.Elements():
            vertices = [mesh[v].point for v in el.vertices]
            a = edge_length(vertices[0], vertices[1], 2)
            b = edge_length(vertices[1], vertices[2], 2)
            c = edge_length(vertices[2], vertices[0], 2)
            circumradius = circumradius_2D(a, b, c)
            max_h = max(max_h, circumradius)
    
    elif mesh.dim == 3:
        for el in mesh.Elements():
            vertices = [mesh[v].point for v in el.vertices]
            circumradius = circumradius_3D(vertices)
            max_h = max(max_h, circumradius)
    
    return max_h

#Calculate Errors
def L2errorVOL(u, u_h, mesh, c=1):
    """Function to compute the L2 error in the volume, c=1 by default"""
    errorVOL = Integrate(c*(u - u_h)**2*dx, mesh)
    return errorVOL
def L2errorBND(u, u_h, mesh, c=1):
    """Function to compute the L2 error on the boundary with skeleton=True, c=1 by default"""
    dS = ds(skeleton =True, definedon=mesh.Boundaries(".*"))
    errorBND = Integrate(c*(u - u_h)**2*dS, mesh)
    return errorBND   
# Create Geometry function
def createGeometry(n):
    structuredMeshUnitBrick = MakeStructured3DMesh(hexes=False, nx=n, ny=n, nz=n)
    return structuredMeshUnitBrick

# Hodge Laplace for 1-forms function
useGMRes = True

def hodgeLaplace1Forms(mesh,
                       g = None, # this is the manufactured solution, when none is given we set it to the zero solution
                       order = 1,
                       C_w = 1):
    
    if g is None:
        g = CF((0,0,0))

    H_curl = HCurl(mesh, order=order, type1=True)  # For 1-forms, H(curl) space
    H_1 = H1(mesh, order=order)     # For 0-forms, H1 space
    fes = H_1 * H_curl
    (p, u), (q, v) = fes.TnT()

    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    t = specialcf.tangential(mesh.dim)
    dS = ds(skeleton=True, definedon=mesh.Boundaries(".*"))

    f = CF(GCurl(GCurl(g)) - GGrad(GDiv(g)))                             
        
    gamma_n_u = Cross(n, curl(u))
    gamma_n_v = Cross(n, curl(v))

    gamma_p_v = v - n*(v*n)
    gamma_p_u = u - n*(u*n)
    gamma_p_g = g - n*(g*n)

    B, F  = BilinearForm(fes), LinearForm(fes)

    B +=  curl(u) * curl(v) * dx
    B +=  grad(p) * v * dx
    B += u * grad(q) * dx
    B += - p * q * dx

    B += gamma_n_v * gamma_p_u * dS
    B += gamma_p_v * gamma_n_u * dS
    B += (C_w/h) * gamma_p_v * gamma_p_u * dS

    F += f * v * dx
    F +=  (C_w / h) * gamma_p_g * gamma_p_v * dS
    F +=  gamma_n_v * gamma_p_g * dS
    F +=  (g*n) * q * ds
    
    with TaskManager(): 
        if (useGMRes == False):
            B.Assemble()
            F.Assemble()
            sol = GridFunction(fes)
            res = F.vec-B.mat * sol.vec
            inv = B.mat.Inverse(freedofs=fes.FreeDofs(), inverse="pardiso")
            sol.vec.data += inv * res
        else:
            #with TaskManager():
            B.Assemble()
            F.Assemble()
            sol = GridFunction(fes)
            blocks = fes.CreateSmoothingBlocks()
            prebj = B.mat.CreateBlockSmoother(blocks)
            GMRes(A =B.mat,x= sol.vec, b=F.vec,pre = prebj,  printrates="\r", maxsteps = 10000, tol=1e-8)
            res = CF((0,0,0))
    

    
    gf_p , gf_u = sol.components

    # Computation of all quantities needed to derive errors
    curl_u = curl(gf_u)
    grad_p = grad(gf_p)
    curl_g = CF(GCurl(g))
    p_m = - CF(GDiv(g))
    grad_p_m = CF(GGrad(p_m))
    gf_gamma_p_u = CF((gf_u - n*(gf_u*n)))
    gf_gamma_p_g = CF((g - n*(g*n)))
    gf_gamma_n_u = BoundaryFromVolumeCF(curl_u)
    gf_gamma_n_g = BoundaryFromVolumeCF(curl_g)
    gf_u_n = CF(gf_u * n)
    gf_g_n = CF(g * n)
    h_avg = 1 / Integrate(1, mesh, VOL) * Integrate(h, mesh, VOL)
    # Actual error evaluation
    # Computation of L2 errors in the volume
    E_u = L2errorVOL(gf_u, g, mesh)
    E_curl_u = L2errorVOL(curl_u, curl_g, mesh)
    E_H_curl_u = E_u + E_curl_u
    E_p = L2errorVOL(gf_p, p_m, mesh)
    E_grad_p = L2errorVOL(grad_p, grad_p_m, mesh)
    # Computation of L2 errors on the boundary
    E_gamma_p_u = L2errorBND(gf_gamma_p_u, gf_gamma_p_g, mesh)
    E_gamma_n_u = L2errorBND(gf_gamma_n_u, gf_gamma_n_g, mesh)
    E_u_n_Gamma = L2errorBND(gf_u_n, gf_g_n, mesh)
    # Hashtag and X Error norm
    HT_E_gamma_p_u = h_avg**(-1)*E_gamma_p_u
    HT_E_gamma_n_u = h_avg*E_gamma_n_u
    HT_E_u = E_H_curl_u + HT_E_gamma_p_u + HT_E_gamma_n_u
    E_h_grad_p = h_avg**2 * E_grad_p
    X_E_u_p = HT_E_u + E_p + E_h_grad_p
    print(sqrt(E_u))
    return (fes.ndof, Norm(res), 
            sqrt(E_u), sqrt(E_curl_u), sqrt(E_H_curl_u), 
            sqrt(E_p), sqrt(E_grad_p), 
            sqrt(E_gamma_p_u), sqrt(E_u_n_Gamma), 
            sqrt(HT_E_gamma_p_u), sqrt(HT_E_gamma_n_u), sqrt(HT_E_u), sqrt(E_h_grad_p),
            sqrt(X_E_u_p))


saveBigCSV = True

g = CF((x**2 * sin(y) * z, - y**3 * cos(z)* 2*x, cos(x) * z**2 * sin(x) * 1/4*y))

refinementVals = [4, 6, 8, 10]
#refinementVals = 3
Cw_vals = logspace_custom_decades(10**0, 100, 25)
orders = [1, 2, 3, 4]

# Cw_vals = [10]
# refinementVals = [8]
# orders = [1]


maxh_values = [] 
all_results = []

for refinementVal in refinementVals:

    mesh = createGeometry(refinementVal)
    h_max_eval = calc_hmax(mesh)
    maxh_values.append(h_max_eval)
    print("doing h_max: ", h_max_eval)

    for order_cw in orders:
        results_cw = []
        print("doing order: ", order_cw)

        for C_w in Cw_vals:
            ndof, res, E_u, E_curl_u, E_H_curl_u, \
            E_p, E_grad_p, \
            E_gamma_p_u, E_u_n_Gamma, \
            HT_E_gamma_p_u, HT_E_gamma_n_u, HT_E_u, E_h_grad_p, X_E_u_p = hodgeLaplace2Forms(
                mesh, g=g, order=order_cw, C_w=C_w
            )

            row_dict = {
                'order': order_cw,
                'hmax': h_max_eval,
                'C_w': C_w,
                'ndof': ndof,
                'res': res,
                'E_u': E_u,
                'E_curl_u': E_curl_u,
                'E_H_curl_u': E_H_curl_u,
                'E_p': E_p,
                'E_grad_p': E_grad_p,
                'E_gamma_p_u': E_gamma_p_u,
                'E_u_n_Gamma': E_u_n_Gamma,
                'HT_E_gamma_p_u': HT_E_gamma_p_u,
                'HT_E_gamma_n_u': HT_E_gamma_n_u,
                'HT_E_u': HT_E_u,
                'E_h_grad_p': E_h_grad_p,
                'X_E_u_p': X_E_u_p
            }
            all_results.append(row_dict)

df_all_results = pd.DataFrame(all_results)
df_all_results.to_csv("all_3D_2forms_simulation_results.csv", index=False)

print("Successfully ran all combinations.")
