import numpy as np
import pydot
import time
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from pydrake.geometry.optimization import (
    HPolyhedron,
    Point,
)
from pydrake.math import (
    BsplineBasis,
    BsplineBasis_,
    KnotVectorType,
)
from pydrake.solvers import(
    MosekSolver,
    IpoptSolver,
    SnoptSolver,

    MathematicalProgram,
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    QuadraticCost,
    PerspectiveQuadraticCost,
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

from gcs.base import BaseGCS
from gcs.bezier import BezierTrajectory

from helpers import plot_bezier

def single_spline_opt():
    order = 4
    num_basis_functions = 4
    # start and end control point
    zeros = np.array([[0.0],
                      [0.0]])
    infs = np.array([[np.inf],
                     [np.inf]])
    
    x0 = np.array([[0.0], 
                   [0.0]])
    xf = np.array([[1.0], 
                   [2.0]])
    dx0 = np.array([[0.2],
                    [-0.5]])
    dxf = np.array([[0.0],
                    [0.0]])
    
    t0 = np.array([[0.0]])
    tf = np.array([[1.0]])
    eta = 0.25
    
    rlb0 = np.array([[-1],
                     [-1]])
    rub0 = np.array([[1],
                     [2]])
    drlb0 = np.array([[-1],
                      [-1]])
    drub0 = np.array([[1],
                      [2]])
    ddrlb0 = np.array([[-2],
                       [-2]])
    ddrub0 = np.array([[2],
                       [2]])
    

    
    prog = MathematicalProgram()
    rs = prog.NewContinuousVariables(2,4)
    hs = prog.NewContinuousVariables(1,4)
    vars = np.concatenate((rs.flatten("F"),hs.flatten("F")))

    # create the trajectory Bezier
    r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order, num_basis_functions, 
                                      KnotVectorType.kClampedUniform, 0., 1.),rs)
    h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order, num_basis_functions, 
                                      KnotVectorType.kClampedUniform, 0., 1.),hs)
    

    r_control_points = r_trajectory.control_points()
    h_control_points = h_trajectory.control_points()
    dr_control_points = r_trajectory.MakeDerivative(1).control_points()
    dh_control_points = h_trajectory.MakeDerivative(1).control_points()
    ddr_control_points = r_trajectory.MakeDerivative(2).control_points()
    ddh_control_points = h_trajectory.MakeDerivative(2).control_points()

    print(len(r_control_points))
    print(len(dr_control_points))
    print(len(ddr_control_points))


    ### PATH
    ## constrain the initial control point
    prog.AddLinearEqualityConstraint(
        DecomposeLinearExpressions(r_control_points[0][0:2],vars),x0[0:2],vars)
    prog.AddLinearEqualityConstraint(
        DecomposeLinearExpressions(dr_control_points[0][0:2],vars) - 
        DecomposeLinearExpressions(dh_control_points[0][0:2],vars)*dx0[0:2],zeros,vars)
    ## constrain the final control point
    prog.AddLinearEqualityConstraint(
        DecomposeLinearExpressions(r_control_points[-1][0:2],vars),xf[0:2],vars)
    prog.AddLinearEqualityConstraint(
        DecomposeLinearExpressions(dr_control_points[-1][0:2],vars) - 
        DecomposeLinearExpressions(dh_control_points[-1][0:2],vars)*dxf[0:2],zeros,vars)
    ## control points of 0th derivative in a convex set around x: [-1, 1], y: [-2, 2]
    for i in range(1,order-1):
        prog.AddLinearConstraint(
            DecomposeLinearExpressions(r_control_points[i][0:2],vars),rlb0[0:2],rub0[0:2],vars)
    ## control points of 1st derivative in a convex set around x: [-1, 1]*dh(s), y: [-2, 2]*dh(s)
    for i in range(0,len(dr_control_points)):
        prog.AddLinearConstraint(
            DecomposeLinearExpressions(dr_control_points[i][0:2],vars) - 
            drub0[0:2]*DecomposeLinearExpressions(dh_control_points[i][0:2],vars),-infs,zeros,vars)
        prog.AddLinearConstraint(
            DecomposeLinearExpressions(dr_control_points[i][0:2],vars) - 
            drlb0[0:2]*DecomposeLinearExpressions(dh_control_points[i][0:2],vars),zeros,infs,vars)
    ## control points of 2nd derivative in a convex set of actuator constraints
    # TODO: this is nonconvex because simult. opt. r and h, not sure why
    # for i in range(0,len(ddr_control_points)):
    #     prog.AddLinearConstraint(
    #         DecomposeLinearExpressions(ddr_control_points[i][0:2],vars) -
    #         ddrub0[0:2]*DecomposeLinearExpressions(ddh_control_points[i][0:2],vars),-infs,zeros,vars)
    #     prog.AddLinearConstraint(
    #         DecomposeLinearExpressions(ddr_control_points[i][0:2],vars) -
    #         ddrlb0[0:2]*DecomposeLinearExpressions(ddh_control_points[i][0:2],vars),zeros,infs,vars)
        
    ### TIME
    ## constrain the initial control point
    prog.AddLinearEqualityConstraint(
        DecomposeLinearExpressions(h_control_points[0][0],vars),t0[0],vars)
    # ## constrain the final control point
    # prog.AddLinearConstraint(
    #     DecomposeLinearExpressions(h_control_points[-1][0],vars),tf[0]-eta,tf[0]+eta,vars)
    ## constrain the final control point
    prog.AddLinearEqualityConstraint(
        DecomposeLinearExpressions(h_control_points[-1][0],vars),tf[0],vars)
    ## control points for the 1st derivative to be strictly positive
    for i in range(0,len(dh_control_points)):
        prog.AddLinearConstraint(
            DecomposeLinearExpressions(dh_control_points[i][0],vars),np.array([0.0]),np.array([np.inf]),vars)

    ### COST
    ## final time 
    a = 1
    if a != 0:
        hS = h_control_points[-1] - h_control_points[0]
        prog.AddQuadraticCost(vars[-1])

    ## path length
    b = 0
    if b != 0:
        for i in range(len(dr_control_points)):
            H = DecomposeLinearExpressions(dr_control_points[i]/order, vars)
            prog.AddL2NormCost(H,np.zeros(2),vars)

    ## path integral
    c = 1
    # TODO


    solver = IpoptSolver()
    t0 = time.time()
    result = solver.Solve(prog)
    print("Solving took: ", time.time()-t0)

    rs_sol = result.GetSolution(rs)
    hs_sol = result.GetSolution(hs)
    print("rs: ", rs_sol)
    print("hs: ", hs_sol)

    plot_bezier(rs_sol,hs_sol)


if __name__ == "__main__":
    # initial_testing()
    single_spline_opt()

    