import os
import shutil
import pickle
import time
import pprint
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from IPython.display import SVG

from pydrake.examples import QuadrotorGeometry
from pydrake.geometry import MeshcatVisualizer, Rgba, StartMeshcat
from pydrake.geometry.optimization import HPolyhedron, VPolytope
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.perception import PointCloud
from pydrake.solvers import GurobiSolver,  MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

from gcs.bezier import BezierGCS
from reproduction.uav.helpers import FlatnessInverter
from reproduction.uav.building_generation import *
from reproduction.util import *

g_lic = GurobiSolver.AcquireLicense()
m_lic = MosekSolver.AcquireLicense()



# Start the visualizer (run this cell only once, each instance consumes a port)
meshcat = StartMeshcat()

meshcat.SetProperty("/Grid", "visible", False)
meshcat.SetProperty("/Axes", "visible", False)
meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)



start = np.array([-1, -1])
goal = np.array([2, 1])
building_shape = (3, 3)
start_pose = np.r_[(start-start)*5, 1.]
goal_pose = np.r_[(goal-start)*5., 1.]

# Generate a random building
np.random.seed(3)
grid, outdoor_edges, wall_edges = generate_grid_world(shape=building_shape, start=start, goal=goal)
regions = compile_sdf(FindModelFile("models/room_gen/building.sdf"), grid, start, goal, outdoor_edges, wall_edges)



# Build the GCS optimization
gcs = BezierGCS(regions, order=7, continuity=4, hdot_min=1e-3, full_dim_overlap=True)

gcs.addTimeCost(1e-3)
gcs.addPathLengthCost(1)
gcs.addVelocityLimits(-10 * np.ones(3), 10 * np.ones(3))
regularization = 1e-3
gcs.addDerivativeRegularization(regularization, regularization, 2)
gcs.addDerivativeRegularization(regularization, regularization, 3)
gcs.addDerivativeRegularization(regularization, regularization, 4)
gcs.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)

gcs.setPaperSolverOptions()
gcs.setSolver(MosekSolver())

# Solve GCS
trajectory = gcs.SolvePath(True, verbose=False, preprocessing=True)[0]



view_regions = False
track_uav = False

# Build and run Diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

parser = Parser(plant, scene_graph)
parser.package_map().Add("gcs", GcsDir())
model_id = parser.AddModelFromFile(FindModelFile("models/room_gen/building.sdf"))

plant.Finalize()

meshcat_cpp = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)


animator = meshcat_cpp.StartRecording()
if not track_uav:
    animator = None
traj_system = builder.AddSystem(FlatnessInverter(trajectory, animator))
quad = QuadrotorGeometry.AddToBuilder(builder, traj_system.get_output_port(0), scene_graph)

diagram = builder.Build()

# Set up a simulator to run this diagram
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)

meshcat.Delete()

if view_regions:
    for ii in range(len(regions)):
        v = VPolytope(regions[ii])
        meshcat.SetTriangleMesh("iris/region_" + str(ii), v.vertices(),
                                ConvexHull(v.vertices().T).simplices.T, Rgba(0.698, 0.67, 1, 0.4))
        
# Simulate
end_time = trajectory.end_time()
simulator.AdvanceTo(end_time+0.05)
meshcat_cpp.PublishRecording()




with open (os.path.join(GcsDir(), "data/uav_example/trajectory.html"), "w") as f:
    f.write(meshcat.StaticHtml())


