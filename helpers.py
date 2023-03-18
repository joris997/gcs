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


def get_derivative_control_points(control_points):
    # get the control points of the derivative dr of the curve r 
    dimension = control_points.shape[0]
    order = control_points.shape[1]

    dcontrol_points = np.zeros((dimension,order-1))
    for i in range(order-1):
        dcontrol_points[:,i] = order*(control_points[:,i+1]-control_points[:,i])
    return dcontrol_points

def plot_bezier(r_control_points,h_control_points):
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
    
    nevals = 25
    evals = np.linspace(0,1,nevals)
    r_eval = np.zeros((2,nevals))
    h_eval = np.zeros((1,nevals))
    dr_eval = np.zeros((2,nevals))
    dh_eval = np.zeros((1,nevals))
    drdt_eval = np.zeros((2,nevals))
    
    for idx,eval in enumerate(evals):
        r_eval[:,idx] = r_bspline.EvaluateCurve(r_control_points.T,eval)
        h_eval[:,idx] = h_bspline.EvaluateCurve(h_control_points.T,eval)

        dr_eval[:,idx] = dr_bspline.EvaluateCurve(dr_control_points.T,eval)
        dh_eval[:,idx] = dh_bspline.EvaluateCurve(dh_control_points.T,eval)

        drdt_eval[:,idx] = dh_bspline.EvaluateCurve(dh_control_points.T,eval) * \
                            dr_bspline.EvaluateCurve(dr_control_points.T,eval)

    fig, axs = plt.subplots(2,2)

    ### position
    axs[0,0].scatter(r_eval[0,:],r_eval[1,:],color='k')
    axs[0,0].set_xlabel('x position')
    axs[0,0].set_ylabel('y position')
    axs[0,0].set_title('position')
    axs[0,0].set_aspect('equal')

    axs[0,1].scatter(evals,h_eval,color='k')
    axs[0,1].set_xlabel('time')
    axs[0,1].set_ylabel('phase')
    axs[0,1].set_title('position')

    ### velocity
    # for idx,element in enumerate(h_eval):
    axs[1,0].scatter(h_eval,drdt_eval[0,:],color='g')
    axs[1,0].scatter(h_eval,drdt_eval[1,:],color='k')
    axs[1,0].set_xlabel('time')
    axs[1,0].set_ylabel('vel')
    axs[1,0].set_title('velocity')

    axs[1,1].scatter(evals,dh_eval,color='k')
    axs[1,1].set_xlabel('dtime')
    axs[1,1].set_ylabel('phase')
    axs[1,1].set_title('velocity')
    
    plt.show()
