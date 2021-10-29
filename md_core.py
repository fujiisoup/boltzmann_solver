"""
molecular dynamics simulation but with point-like atom approximation
"""
import numpy as np
from .core import scattering


def get_colliding_time(x_rel, v_rel, collision_flag=None):
    """
    Find the colliding time, which is the time to the nearest approach, 
    based on relative position and velocities

    collision_flag: True if the two particle will collide. 
        It should be computed based on the collision crosssection 
    """
    vsq = -np.sum(v_rel * v_rel, axis=-1, keepdims=True)
    v_inv = v_rel / vsq  # inverse of the velocity but with the same direction
    colliding_time = np.sum(x_rel * v_inv, axis=-1)
    return np.where(colliding_time > 0, colliding_time, np.inf)


class LargeWall:
    """
    Walls with the infinite size.
    """
    def __init__(self, x, n, restitution_coef=None):
        """
        x: [n_walls, dimension]
            center position
        n: [n_walls, dimension]
            normal vector to the walls
        """
        assert (x.ndim == 2)
        assert (n.ndim == 2)
        self.x = x
        self.n = n
        if restitution_coef is None:
            restitution_coef = np.ones(len(self.x))
        self.restitution_coef = restitution_coef

    def get_colliding_time(self, x, v):
        """
        Estimate the colliding time dependence for particles having x and v

        x: [n_particles, dimension] or [dimension]
        v: [n_particles, dimension] or [dimension]
        """
        v_rel = np.tensordot(self.n, v, axes=(-1, -1))  # [n_walls, n_particles] or [n_walls]
        if x.ndim == 2:
            x_rel = np.sum((self.x[:, np.newaxis] - x) * self.n, axis=-1)  # [n_walls, n_particles]
        else:  # x.ndim == 1
            x_rel = np.sum((self.x - x) * self.n, axis=-1)  # [n_walls]

        vsq = -np.sum(v_rel * v_rel, axis=-1, keepdims=True)
        v_inv = v_rel / vsq  # inverse of the velocity but with the same direction

        colliding_time = np.matmul(x_rel[..., np.newaxis], v_inv[..., np.newaxis, :])
        return np.where(colliding_time > 0, colliding_time, np.inf)

    def reflect(self, i, v):
        """ Make a reflection of particles by walls

        Parameters
        ----------
        i: wall index
        v: velocity of particle
        """
        v_ref = np.dot(v @ self.n[i])
        return v + 2 * v_ref * self.n[i]


class Particles:
    """
    Class for point-like particles
    """
    def __init__(self, x, v, wall):
        self.x = x  # shape [particle index, dimension]
        self.v = v
        x_rel = self.x - self.x[:, np.newaxis]
        v_rel = self.v - self.v[:, np.newaxis]
        self.colliding_time = get_colliding_time(x_rel, v_rel)

        # collision with walls
        self.wall = wall
        self.wall_colliding_time = wall.get_colliding_time(x, v)
        self.accum_time = 0

    def update_v(self, i, v):
        """
        update velocities changed by collision 
        
        Parameters
        ----------
        i: particle index to be update
        v: new velocities [dimension]
        """
        # collision time among particles
        self.v[i] = v
        x_rel = self.x[i] - self.x
        v_rel = self.v[i] - self.v
        colliding_time = get_colliding_time(x_rel, v_rel) + self.accum_time
        self.colliding_time[i] = colliding_time
        self.colliding_time[:, i] = colliding_time
        # collision with walls
        self.wall_colliding_time[:, i] = self.wall.get_colliding_time(self.x, v) + self.accum_time

    def update_x(self, dt):
        # update the position
        self.accum_time += dt
        self.x += self.v * dt

    def run1step(self):
        # find the smallest dt
        ip, jp = np.unravel_index(np.argmin(self.colliding_time), self.colliding_time.shape)
        t_next = self.colliding_time[ip, jp]
        
        iw, jw = np.unravel_index(np.argmin(self.wall_colliding_time), self.wall_colliding_time.shape)
        t_next_wall = self.wall_colliding_time[iw, jw]

        if t_next < t_next_wall:
            dt = t_next - self.accum_time
            self.update_x(dt)
            # make a collision here
            raise NotImplementedError

        else:  # t_next_wall < t_next:
            dt = t_next_wall - self.accum_time
            self.update_x(dt)
            # make a collision with wall here
            raise NotImplementedError


def choose_colliding_particles(x, v):
    """
    Choose a pair of coliding particles

    Returns an index (i, j) and time to colide.
    """
    relative_x = x - x[:, np.newaxis]
    relative_v = v - v[:, np.newaxis]
    coliding_time = relative_x / relative_v
    coliding_time = np.where(coliding_time > 0, coliding_time, np.inf)
    coliding_time = np.min(coliding_time, axis=-1)
    # find the colliding particle pairs in the nearest future
    print(coliding_time)
    index = np.unravel_index(np.argmin(coliding_time), shape=coliding_time.shape)
    return index
