"""
molecular dynamics simulation but with point-like atom approximation
"""
import numpy as np
from .core import scattering


def get_colliding_time(x_rel, v_rel, radius=None):
    """
    Find the colliding time, which is the time to the nearest approach, 
    based on relative position and velocities

    radius: radius of particles.
        Should be a float or array of floats
        If radius is given, also returns the square of the normalized impact parameter
    """
    vsq = -np.sum(v_rel * v_rel, axis=-1, keepdims=True)
    v_inv = v_rel / vsq  # inverse of the velocity but with the same direction
    colliding_time = np.sum(x_rel * v_inv, axis=-1)
    
    colliding_time = np.where(colliding_time > 0, colliding_time, np.inf)
    if radius is None:
        return colliding_time

    # compute impact parameters
    dx = x_rel + v_rel * colliding_time[:, np.newaxis]
    impact_parameter2 = np.sum(dx**2, axis=-1) / radius**2
    return np.where(impact_parameter2 <= 1.0, colliding_time, np.inf)
    

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
        return v + (1 + self.restitution_coef[i]) * v_ref * self.n[i]


class Particles:
    """
    Class for point-like particles
    """
    def __init__(self, x, v, wall, radius):
        self.x = x  # shape [particle index, dimension]
        self.v = v
        self._initialize()

        # collision with walls
        self.wall = wall
        self.wall_colliding_time = wall.get_colliding_time(x, v)
        self.accum_time = 0
        self._radius = radius

    def _initialize(self):
        x_rel = self.x - self.x[:, np.newaxis]
        v_rel = self.v - self.v[:, np.newaxis]
        self.colliding_time = get_colliding_time(x_rel, v_rel, self.radius(v_rel))

    def radius(self, v_rel):
        return self._radius

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
        colliding_time = get_colliding_time(x_rel, v_rel, self.radius(v_rel))
        colliding_time = colliding_time + self.accum_time
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
            v1, v2 = self.collision(
                self.x[ip], self.x[jp], self.v[ip], self.v[jp]
            )
            self.v[ip], self.v[jp] = v1, v2

        else:  # t_next_wall < t_next:  wall collision
            dt = t_next_wall - self.accum_time
            self.update_x(dt)
            # make a collision with wall here
            self.v[iw] = self.wall.reflect([jw], self.v[iw])

    def collision(self, x1, x2, v1, v2):
        """
        Compute the post-collision velocities of particle 1 and 2.
        
        Parameters
        ----------
        x1, x2: positions of the particle 1 and 2
        v1, v2: velocities of the particle 1 and 2
        
        Returns
        -------
        v1, v2: new velocities of the particle 1 and 2
        """
        return v1, v2