"""AVLite-native lattice display compatibility, with no GUI imports."""


def edge_trajectory(edge):
    from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker
    trajectory = TrajectoryTracker(path=[tuple(p) for p in edge.xy], velocity=[0.0]*len(edge.xy))
    trajectory.path_s_from_parent = edge.s.tolist()
    trajectory.path_d_from_parent = edge.d.tolist()
    return trajectory
