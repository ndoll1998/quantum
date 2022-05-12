import quantum
import numpy as np
import open3d as o3d
from functools import partial
from itertools import cycle

if __name__ == '__main__':
    
    # create the wave function and mechanics instance
    wf = quantum.analytic.HydrogenLike(5, 2, 1)
    mechanics = quantum.extra.BohmianMechanics(wave=wf)

    # sample points from wave function distribution
    q = quantum.utils.sampling.inverse_transfer_sampling(
        pdf=partial(wf.pdf, t=0),
        num_samples=10_000,
        bounds=[[0, 40], [0, 2*np.pi], [0, np.pi]],
        delta=0.05
    )

    # covert to cartesian coordinate system
    x = quantum.utils.spherical.spherical_to_cartesian(q[:, 0], q[:, 1], q[:, 2])
    x = np.stack(x, axis=-1)
    # create pointcloud from sampled points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)

    # create simulation iterator
    simulation = mechanics.simulate(q, t0=0, dt=0.1)

    def simulate(vis):
        _, q = next(simulation)
        # covert to cartesian coordinate system
        x = quantum.utils.spherical.spherical_to_cartesian(q[:, 0], q[:, 1], q[:, 2])
        x = np.stack(x, axis=-1)
        # update geometry
        vis.update_geometry()
        pcd.points = o3d.utility.Vector3dVector(x)

    # visualize
    vis = o3d.visualization.draw_geometries_with_animation_callback(
        [pcd], simulate
    )