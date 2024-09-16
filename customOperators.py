from ngsolve import *
from ngsolve.fem import LeviCivitaSymbol, Einsum
coords = [x,y,z]

def JacobianOfCF(cf , dim):
    """ Function to compute the Jacobi Matrix of a vector coefficient function cf """

    if dim == 3:
        Jac_u_3D = CF((
        cf[0].Diff(x), cf[0].Diff(y), cf[0].Diff(z),
        cf[1].Diff(x), cf[1].Diff(y), cf[1].Diff(z),
        cf[2].Diff(x), cf[2].Diff(y), cf[2].Diff(z)
        ), dims=(3, 3))

        return Jac_u_3D
    
    elif dim == 2:
        Jac_u_2D = CF((
        cf[0].Diff(x), cf[0].Diff(y),
        cf[1].Diff(x), cf[1].Diff(y)
        ), dims=(2, 2))

        return Jac_u_2D
    
    else:
        print("no valid mesh dimension found..")
        return None

def GGrad(cf, dim):
    """ Function to compute the gradient of a scalar Coefficient Function """
    gg = [cf.Diff(coords[i]) for i in range(dim)]
    return CF(tuple(gg))


def GCurl(cf, dim):
    """ Function to compute the curl of vec cf using Jacobian """
    
    if dim == 3:
        Jac_u = JacobianOfCF(cf, dim)
        curl_u = CF((Jac_u[2,1] - Jac_u[1,2],  
                    Jac_u[0,2] - Jac_u[2,0],  
                    Jac_u[1,0] - Jac_u[0,1]))
    
        return curl_u
    
    elif dim == 2:
        Jac_u = JacobianOfCF(cf)
        curl_u = CF((Jac_u[1,0] - Jac_u[0,1],  
                    Jac_u[0,1] - Jac_u[1,0] ))
        
        return curl_u

def GDiv(cf, dim):
    """ Function to compute the divergence of a vector coefficient function """

    gd = [cf[i].Diff(coords[i]) for i in range(dim)]
    
    return CF(sum(gd))