import numpy as np
import pydot
import time
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from pydrake.math import (
    BsplineBasis,
    BsplineBasis_,
    KnotVectorType,
)
from pydrake.symbolic import (
    DecomposeLinearExpressions,
    Expression,
    MakeMatrixContinuousVariable,
    MakeVectorContinuousVariable,
)
from pydrake.trajectories import (
    BsplineTrajectory,
    BsplineTrajectory_,
    Trajectory,
)

class Polygon():
    def __init__(self,A,b,eps):
        self.A = A
        self.b = b
        self.eps = eps # already in b, but for determining plotting ranges

        self.sides = self.A.shape[0]
        
def wrap(theta, lower, upper):
    assert(lower < upper)
    return (theta - lower) % (upper - lower) + lower

def get_halfspace_polyhedral(x0,xf,eps):
    # x0 and xf are 2x1 numpy arrays
    # eps is a scalar indicating the permissible range around x0 and xf
    # returns A and b of Ax <= b
    assert(x0.shape == (2,1))
    assert(xf.shape == (2,1))
    assert(eps > 0)

    dx = xf[0] - x0[0]
    dy = xf[1] - x0[1]
    dx = dx[0]
    dy = dy[0]

    print("dx : ", dx, "   dy : ", dy)

    # scenario 1: horizontal line
    if abs(dy) < 1e-6:
        y = x0[1]
        A = np.array([[-1., 0],
                      [1., 0],
                      [0, -1.],
                      [0, 1.]])
        b = np.array([-x0[0] + eps,
                      xf[0] + eps,
                      -y + eps,
                      y + eps])
    # scenario 2: vertical line
    elif abs(dx) < 1e-6:
        x = x0[0]
        A = np.array([[-1., 0],
                      [1., 0],
                      [0, -1.],
                      [0, 1.]])
        b = np.array([-x + eps,
                      x + eps,
                      -x0[1] + eps,
                      xf[1] + eps])
    # scenario 3: general line
    else:
        th = np.arctan2(dy,dx)
        th_ = wrap(th,0,np.pi)
        deps = eps/np.cos(th_)
        # lower y
        a1 = np.array([dy/(dx), -1.])
        bnd = deps if dy < 0 else -deps
        b1 = a1[0]*x0[0] + a1[1]*x0[1] + bnd
        # higher y
        a2 =  np.array([dy/(dx), -1.])
        bnd = -deps if dy < 0 else deps
        b2 =  a2[0]*xf[0] + a2[1]*xf[1] + bnd

        deps = eps/np.sin(th_)
        # lower x
        a3 = np.array([-1/a1[0], -1.])
        bnd = -deps if dy < 0 else deps
        b3 = a3[0]*x0[0] + a3[1]*x0[1] + bnd
        # higher x
        a4 =  np.array([-1/a2[0], -1.])
        bnd = deps if dy < 0 else -deps
        b4 = a4[0]*xf[0] + a4[1]*xf[1] + bnd

        A = np.zeros((4,2))
        b = np.zeros((4,))
        if dx > 0:
            A[0,:] = -a1
            A[1,:] = a2
            b[0] = -b1
            b[1] = b2
        else:
            A[0,:] = a1
            A[1,:] = -a2
            b[0] = b1
            b[1] = -b2
        
        if dy > 0:
            A[2,:] = a3
            A[3,:] = -a4
            b[2] = b3
            b[3] = -b4
        else:
            A[2,:] = -a3
            A[3,:] = a4
            b[2] = -b3
            b[3] = b4

    return A,b


def get_derivative_control_points(control_points):
    # get the control points of the derivative dr of the curve r 
    dimension = control_points.shape[0]
    order = control_points.shape[1]

    dcontrol_points = np.zeros((dimension,order-1))
    for i in range(order-1):
        dcontrol_points[:,i] = order*(control_points[:,i+1]-control_points[:,i])
    return dcontrol_points

def plot_bezier(r_control_points,h_control_points,polygons):
    r_order = r_control_points.shape[1]
    h_order = h_control_points.shape[1]
    r_num_basis_functions = r_control_points.shape[1]
    h_num_basis_functions = h_control_points.shape[1]

    r_bspline = BsplineBasis(r_order,r_num_basis_functions,
                             KnotVectorType.kClampedUniform,0.,1.)
    h_bspline = BsplineBasis(h_order,h_num_basis_functions,
                             KnotVectorType.kClampedUniform,0.,1.)
    
    dr_control_points = get_derivative_control_points(r_control_points)
    dh_control_points = get_derivative_control_points(h_control_points)
    dr_bspline = BsplineBasis(r_order-1,r_num_basis_functions-1,
                              KnotVectorType.kClampedUniform,0.,1.)
    dh_bspline = BsplineBasis(h_order-1,h_num_basis_functions-1,
                              KnotVectorType.kClampedUniform,0.,1.)
    
    nevals = 60
    evals = np.linspace(0,1,nevals)
    r_eval = np.zeros((2,nevals))
    h_eval = np.zeros((1,nevals))
    dr_eval = np.zeros((2,nevals))
    dh_eval = np.zeros((1,nevals))
    drdt_eval = np.zeros((2,nevals))

    q_eval = np.zeros((2,nevals))
    dq_eval = np.zeros((2,nevals))
    
    for idx,eval in enumerate(evals):
        r_eval[:,idx] = r_bspline.EvaluateCurve(r_control_points.T,eval)
        h_eval[:,idx] = h_bspline.EvaluateCurve(h_control_points.T,eval)

        dr_eval[:,idx] = dr_bspline.EvaluateCurve(dr_control_points.T,eval)
        dh_eval[:,idx] = dh_bspline.EvaluateCurve(dh_control_points.T,eval)

        dq_eval[:,idx] = dr_eval[:,idx] / dh_eval[:,idx]

        drdt_eval[:,idx] = dh_bspline.EvaluateCurve(dh_control_points.T,eval) * \
                            dr_bspline.EvaluateCurve(dr_control_points.T,eval)

    fig, axs = plt.subplots(2,2)

    ### position
    axs[0,0].plot(r_eval[0,:],r_eval[1,:],color='k')
    for p in polygons:
        x_range = np.linspace(r_eval[0,:]-p.eps,r_eval[0,:]+p.eps,100)
        for side in range(p.sides):
            y_range = (p.b[side] - p.A[side,0]*x_range)/p.A[side,1]
            axs[0,0].plot(x_range,y_range,color='r')



    axs[0,0].grid(True)
    axs[0,0].set_xlabel('x position')
    axs[0,0].set_ylabel('y position')
    axs[0,0].set_title('position')
    axs[0,0].set_aspect('equal')

    axs[0,1].plot(evals,h_eval[0,:],color='k')
    axs[0,1].grid(True)
    axs[0,1].set_xlabel('time')
    axs[0,1].set_ylabel('phase')
    axs[0,1].set_title('position')

    ### velocity
    # for idx,element in enumerate(h_eval):
    axs[1,0].plot(h_eval[0,:],dq_eval[0,:],color='g',label='dx/dt')
    axs[1,0].plot(h_eval[0,:],dq_eval[1,:],color='k',label='dy/dt')
    axs[1,0].grid(True)
    axs[1,0].set_xlabel('time')
    axs[1,0].set_ylabel('vel')
    axs[1,0].set_title('velocity')
    axs[1,0].legend()

    axs[1,1].plot(evals,dh_eval[0,:],color='k')
    axs[1,1].grid(True)
    axs[1,1].set_xlabel('dtime')
    axs[1,1].set_ylabel('phase')
    axs[1,1].set_title('velocity')
    
    plt.show()
