"""
2D lid-driven cavity solver using vorticity-streamfunction formulation.
Solves: ∂ω/∂t + u·∇ω = (1/Re)∇²ω,  ∇²ψ = -ω,  u = ∂ψ/∂y, v = -∂ψ/∂x.

Boundary conditions:
- Top wall moves with velocity U_lid (lid-driven).
- Other walls are no-slip (u=v=0).
- Domain: unit square [0,1] x [0,1].
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

class LidDrivenCavity:
    """
    Solver for the lid-driven cavity flow.
    """

    def __init__(self, nx, ny, Re, U_lid=1.0, dt=None, nt=5000,
                 omega_relax=1.5, tol=1e-6, max_iter=10000):
        """
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions.
        Re : float
            Reynolds number.
        U_lid : float
            Velocity of the top moving lid.
        dt : float, optional
            Time step. If None, computed from CFL condition.
        nt : int
            Number of time steps.
        omega_relax : float
            Over-relaxation parameter for SOR (1 < omega < 2).
        tol : float
            Tolerance for SOR convergence.
        max_iter : int
            Maximum SOR iterations.
        """
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.U_lid = U_lid
        self.omega_relax = omega_relax
        self.tol = tol
        self.max_iter = max_iter
        self.nt = nt

        # Grid spacing (unit square)
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)

        # Compute stable time step if not provided (CFL + diffusion)
        if dt is None:
            # CFL: dt < min(dx, dy) / U_max
            # Diffusion: dt < 0.5 * Re * min(dx,dy)^2  (explicit diffusion)
            # We take a conservative fraction.
            dmin = min(self.dx, self.dy)
            cfl = 0.5 * dmin / U_lid if U_lid > 0 else 1e-3
            diff = 0.25 * Re * dmin**2
            self.dt = 0.8 * min(cfl, diff)
        else:
            self.dt = dt

        # Initialize fields
        self.psi = np.zeros((ny, nx))      # streamfunction
        self.omega = np.zeros((ny, nx))    # vorticity

        # Store history for plotting
        self.history = {'psi': [], 'omega': [], 'time': []}

    def _sor_poisson(self, rhs):
        """
        Solve ∇²ψ = rhs using Successive Over-Relaxation.
        Returns the solution ψ.
        """
        psi = np.zeros_like(rhs)
        dx2 = self.dx**2
        dy2 = self.dy**2
        factor = self.omega_relax / (2.0*(1.0/dx2 + 1.0/dy2))

        for iteration in range(self.max_iter):
            psi_old = psi.copy()
            # Update interior points
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    psi[i, j] = (1.0 - self.omega_relax) * psi[i, j] + \
                                factor * ( (psi[i+1, j] + psi[i-1, j]) / dx2 +
                                           (psi[i, j+1] + psi[i, j-1]) / dy2 -
                                           rhs[i, j] )
            # Apply Dirichlet boundary conditions (psi = 0 on all walls)
            psi[0, :] = 0.0
            psi[-1, :] = 0.0
            psi[:, 0] = 0.0
            psi[:, -1] = 0.0

            # Check convergence
            diff = np.max(np.abs(psi - psi_old))
            if diff < self.tol:
                break

        return psi

    def _compute_velocities(self):
        """
        Compute u, v from streamfunction using central differences.
        Returns u (ny x nx) and v (ny x nx).
        """
        u = np.zeros((self.ny, self.nx))
        v = np.zeros((self.ny, self.nx))
        # Interior points
        u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2.0 * self.dy)
        v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2.0 * self.dx)
        # Top lid: u = U_lid (imposed)
        u[-1, :] = self.U_lid
        # Other walls: no-slip (already zero)
        return u, v

    def _apply_vorticity_bc(self):
        """
        Set vorticity boundary conditions from streamfunction.
        For no-slip walls: ω = - (2/Δn^2) * (ψ_wall - ψ_adjacent)
        With ψ_wall = 0, so ω = (2/Δn^2) * ψ_adjacent  (note sign depends on normal direction)
        For top moving lid: we need to account for the velocity gradient.
        """
        # Bottom wall (i=0): ω = - (2/dy^2) * (ψ[1,j] - ψ[0,j]) with ψ[0,j]=0
        self.omega[0, :] = - (2.0 / self.dy**2) * self.psi[1, :]
        # Top wall (i=ny-1): moving lid. We need ∂u/∂y at the wall.
        # A common approximation: ω = (2/dy^2)*(ψ[-2,j] - ψ[-1,j]) + (2*U_lid)/dy
        self.omega[-1, :] = - (2.0 / self.dy**2) * self.psi[-2, :] + (2.0 * self.U_lid) / self.dy
        # Left wall (j=0)
        self.omega[:, 0] = - (2.0 / self.dx**2) * self.psi[:, 1]
        # Right wall (j=nx-1)
        self.omega[:, -1] = - (2.0 / self.dx**2) * self.psi[:, -2]

    def step(self):
        """
        Perform one time step.
        """
        # 1. Solve Poisson equation for psi
        self.psi = self._sor_poisson(-self.omega)

        # 2. Compute velocities
        u, v = self._compute_velocities()

        # 3. Compute vorticity boundary conditions (using updated psi)
        self._apply_vorticity_bc()

        # 4. Compute spatial derivatives of omega (central differences)
        domega_dx = np.zeros_like(self.omega)
        domega_dy = np.zeros_like(self.omega)
        d2omega_dx2 = np.zeros_like(self.omega)
        d2omega_dy2 = np.zeros_like(self.omega)

        # Interior points
        domega_dx[1:-1, 1:-1] = (self.omega[1:-1, 2:] - self.omega[1:-1, :-2]) / (2.0 * self.dx)
        domega_dy[1:-1, 1:-1] = (self.omega[2:, 1:-1] - self.omega[:-2, 1:-1]) / (2.0 * self.dy)
        d2omega_dx2[1:-1, 1:-1] = (self.omega[1:-1, 2:] - 2*self.omega[1:-1, 1:-1] + self.omega[1:-1, :-2]) / (self.dx**2)
        d2omega_dy2[1:-1, 1:-1] = (self.omega[2:, 1:-1] - 2*self.omega[1:-1, 1:-1] + self.omega[:-2, 1:-1]) / (self.dy**2)

        # 5. Advection and diffusion terms
        advection = u * domega_dx + v * domega_dy
        diffusion = (1.0 / self.Re) * (d2omega_dx2 + d2omega_dy2)

        # 6. Explicit Euler update: ω_new = ω_old + dt * (-advection + diffusion)
        self.omega[1:-1, 1:-1] += self.dt * (-advection[1:-1, 1:-1] + diffusion[1:-1, 1:-1])

        # 7. Re-apply vorticity boundary conditions (needed after update)
        self._apply_vorticity_bc()

    def solve(self, verbose=True):
        """
        Run the full time-stepping.
        """
        for n in range(self.nt):
            self.step()
            if verbose and n % 500 == 0:
                print(f"Step {n}/{self.nt}, dt={self.dt:.3e}")
            if n % 100 == 0:
                self.history['psi'].append(self.psi.copy())
                self.history['omega'].append(self.omega.copy())
                self.history['time'].append(n * self.dt)
        return self.psi, self.omega

    def plot_streamfunction(self, savefig=False):
        """
        Plot the streamfunction contour.
        """
        X, Y = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(0, 1, self.ny))
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, self.psi, levels=50, cmap=cm.viridis)
        plt.colorbar(label='Streamfunction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Lid-driven cavity, Re={self.Re}, time={self.nt*self.dt:.2f}')
        if savefig:
            plt.savefig('streamfunction.png', dpi=150)
        plt.show()

    def plot_vorticity(self, savefig=False):
        """
        Plot the vorticity field.
        """
        X, Y = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(0, 1, self.ny))
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, self.omega, levels=50, cmap=cm.RdBu_r)
        plt.colorbar(label='Vorticity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Vorticity, Re={self.Re}, time={self.nt*self.dt:.2f}')
        if savefig:
            plt.savefig('vorticity.png', dpi=150)
        plt.show()

# -------------------- Example usage --------------------
if __name__ == "__main__":
    # Parameters
    nx, ny = 65, 65        # grid points
    Re = 100               # Reynolds number
    U_lid = 1.0
    nt = 5000

    # Create solver
    cavity = LidDrivenCavity(nx, ny, Re, U_lid, nt=nt)

    # Run simulation
    start = time.time()
    psi, omega = cavity.solve(verbose=True)
    print(f"Simulation finished in {time.time()-start:.2f} seconds.")

    # Plot results
    cavity.plot_streamfunction(savefig=True)
    cavity.plot_vorticity(savefig=True)