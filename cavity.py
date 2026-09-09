"""
2D lid-driven cavity solver using vorticity-streamfunction formulation.
Optimized with sparse direct solver for the Poisson equation.

Solves:
    ∂ω/∂t + u·∇ω = (1/Re)∇²ω
    ∇²ψ = -ω
    u = ∂ψ/∂y, v = -∂ψ/∂x

Boundary conditions:
    - Top wall: u = U_lid (moving lid)
    - Other walls: no-slip (u = v = 0)
    - ψ = 0 on all walls (streamlines)

Author: Dr. Sapna Makhdoom
Date: September 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from scipy.sparse import diags, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


class LidDrivenCavity:
    """
    Solver for the 2D lid-driven cavity flow using vorticity-streamfunction
    formulation with a sparse direct Poisson solver.
    """

    def __init__(self, nx, ny, Re, U_lid=1.0, dt=None, nt=2000, tol=1e-6):
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
            Time step. If None, computed from CFL and diffusion limits.
        nt : int
            Number of time steps.
        tol : float
            Tolerance (kept for compatibility with sparse solver).
        """
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.U_lid = U_lid
        self.nt = nt
        self.tol = tol

        # Grid spacing (unit square domain)
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)

        # Compute stable time step if not provided
        if dt is None:
            dmin = min(self.dx, self.dy)
            # CFL condition for advection
            cfl = 0.5 * dmin / U_lid if U_lid > 0 else 1e-3
            # Diffusion stability limit
            diff = 0.25 * Re * dmin**2
            self.dt = 0.8 * min(cfl, diff)
        else:
            self.dt = dt

        # Initialize flow fields
        self.psi = np.zeros((ny, nx))      # streamfunction
        self.omega = np.zeros((ny, nx))    # vorticity

        # Build the Poisson matrix once (major optimization!)
        self._build_poisson_matrix()

        # Store history for animation (optional)
        self.history = {'psi': [], 'omega': [], 'time': []}

    def _build_poisson_matrix(self):
        """
        Build the sparse matrix for the Poisson equation ∇²ψ = rhs.
        Uses the 5-point stencil with Dirichlet boundary conditions ψ = 0.
        Matrix is built once and reused for all time steps.
        """
        N = self.nx * self.ny
        dx2 = self.dx**2
        dy2 = self.dy**2

        # Coefficients for the 5-point stencil
        main_diag = -2.0 / dx2 - 2.0 / dy2
        off_x = 1.0 / dx2
        off_y = 1.0 / dy2

        # Build diagonals
        diagonals = [
            main_diag * np.ones(N),      # main diagonal
            off_x * np.ones(N - 1),      # x-direction (left)
            off_x * np.ones(N - 1),      # x-direction (right)
            off_y * np.ones(N - self.nx),  # y-direction (up)
            off_y * np.ones(N - self.nx)   # y-direction (down)
        ]

        offsets = [0, -1, 1, -self.nx, self.nx]

        # Create sparse matrix in LIL format for easy modification
        self.A = lil_matrix(diags(diagonals, offsets, shape=(N, N), format='lil'))

        # Store indices for boundary points
        self.boundary_indices = []

        # Bottom wall (i = 0)
        for j in range(self.nx):
            idx = j
            self.boundary_indices.append(idx)

        # Top wall (i = ny-1)
        for j in range(self.nx):
            idx = (self.ny - 1) * self.nx + j
            self.boundary_indices.append(idx)

        # Left wall (j = 0)
        for i in range(1, self.ny - 1):
            idx = i * self.nx
            self.boundary_indices.append(idx)

        # Right wall (j = nx-1)
        for i in range(1, self.ny - 1):
            idx = i * self.nx + (self.nx - 1)
            self.boundary_indices.append(idx)

        # Apply Dirichlet boundary conditions: set A[i,i] = 1
        for idx in self.boundary_indices:
            self.A[idx, :] = 0
            self.A[idx, idx] = 1

        # Convert to CSR format for fast solving
        self.A_csr = self.A.tocsr()

    def _solve_poisson(self, rhs):
        """
        Solve ∇²ψ = rhs using sparse direct solver.
        rhs is a 2D array (ny x nx) representing -ω.
        Returns ψ (ny x nx).
        """
        N = self.nx * self.ny
        rhs_flat = rhs.flatten()

        # Copy the matrix (we need to modify it for boundary conditions)
        A = self.A_csr.copy()
        b = rhs_flat.copy()

        # Apply Dirichlet boundary conditions: ψ = 0 on all walls
        for idx in self.boundary_indices:
            b[idx] = 0.0

        # Solve the linear system
        psi_flat = spsolve(A, b)
        psi = psi_flat.reshape((self.ny, self.nx))

        return psi

    def _compute_velocities(self):
        """
        Compute u, v from streamfunction using central differences.
        Returns u (ny x nx) and v (ny x nx).
        """
        u = np.zeros((self.ny, self.nx))
        v = np.zeros((self.ny, self.nx))

        # Interior points: central differences
        u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2.0 * self.dy)
        v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2.0 * self.dx)

        # Top lid: u = U_lid (moving wall)
        u[-1, :] = self.U_lid

        return u, v

    def _apply_vorticity_bc(self):
        """
        Set vorticity boundary conditions from streamfunction.
        Uses Thom's formula for no-slip walls.
        """
        # Bottom wall (i = 0)
        self.omega[0, :] = -(2.0 / self.dy**2) * self.psi[1, :]

        # Top wall (i = ny-1) - moving lid
        self.omega[-1, :] = -(2.0 / self.dy**2) * self.psi[-2, :] + (2.0 * self.U_lid) / self.dy

        # Left wall (j = 0)
        self.omega[:, 0] = -(2.0 / self.dx**2) * self.psi[:, 1]

        # Right wall (j = nx-1)
        self.omega[:, -1] = -(2.0 / self.dx**2) * self.psi[:, -2]

    def step(self):
        """
        Perform one time step (explicit Euler).
        """
        # 1. Solve Poisson equation for streamfunction
        self.psi = self._solve_poisson(-self.omega)

        # 2. Compute velocities from streamfunction
        u, v = self._compute_velocities()

        # 3. Apply vorticity boundary conditions
        self._apply_vorticity_bc()

        # 4. Compute spatial derivatives of vorticity
        domega_dx = np.zeros_like(self.omega)
        domega_dy = np.zeros_like(self.omega)
        d2omega_dx2 = np.zeros_like(self.omega)
        d2omega_dy2 = np.zeros_like(self.omega)

        # Interior points: central differences
        domega_dx[1:-1, 1:-1] = (self.omega[1:-1, 2:] - self.omega[1:-1, :-2]) / (2.0 * self.dx)
        domega_dy[1:-1, 1:-1] = (self.omega[2:, 1:-1] - self.omega[:-2, 1:-1]) / (2.0 * self.dy)
        d2omega_dx2[1:-1, 1:-1] = (self.omega[1:-1, 2:] - 2*self.omega[1:-1, 1:-1] + self.omega[1:-1, :-2]) / (self.dx**2)
        d2omega_dy2[1:-1, 1:-1] = (self.omega[2:, 1:-1] - 2*self.omega[1:-1, 1:-1] + self.omega[:-2, 1:-1]) / (self.dy**2)

        # 5. Advection and diffusion terms
        advection = u * domega_dx + v * domega_dy
        diffusion = (1.0 / self.Re) * (d2omega_dx2 + d2omega_dy2)

        # 6. Explicit Euler update: ω_new = ω_old + dt * (-advection + diffusion)
        self.omega[1:-1, 1:-1] += self.dt * (-advection[1:-1, 1:-1] + diffusion[1:-1, 1:-1])

        # 7. Re-apply vorticity boundary conditions
        self._apply_vorticity_bc()

    def solve(self, verbose=True):
        """
        Run the full time-stepping simulation.
        """
        start_time = time.time()

        for n in range(self.nt):
            self.step()

            # Print progress every 200 steps
            if verbose and n % 200 == 0:
                print(f"Step {n}/{self.nt}, dt={self.dt:.3e}")

            # Store history every 50 steps
            if n % 50 == 0:
                self.history['psi'].append(self.psi.copy())
                self.history['omega'].append(self.omega.copy())
                self.history['time'].append(n * self.dt)

        elapsed = time.time() - start_time
        print(f"Simulation finished in {elapsed:.2f} seconds.")
        return self.psi, self.omega

    def plot_streamfunction(self, savefig=False):
        """
        Plot the streamfunction contour.
        """
        X, Y = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(0, 1, self.ny))

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, self.psi, levels=50, cmap=cm.viridis)
        plt.colorbar(contour, label='Streamfunction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Streamfunction, Re={self.Re}, t={self.nt*self.dt:.2f}')

        if savefig:
            plt.savefig('streamfunction.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_vorticity(self, savefig=False):
        """
        Plot the vorticity contour.
        """
        X, Y = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(0, 1, self.ny))

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, self.omega, levels=50, cmap=cm.RdBu_r)
        plt.colorbar(contour, label='Vorticity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Vorticity, Re={self.Re}, t={self.nt*self.dt:.2f}')

        if savefig:
            plt.savefig('vorticity.png', dpi=150, bbox_inches='tight')
        plt.show()


# -------------------- Main Execution --------------------
if __name__ == "__main__":
    # Simulation parameters
    nx, ny = 49, 49    # Grid resolution
    Re = 100           # Reynolds number
    U_lid = 1.0        # Lid velocity
    nt = 2000          # Number of time steps

    # Create and run the solver
    print("Creating solver...")
    cavity = LidDrivenCavity(nx, ny, Re, U_lid, nt=nt)

    print("Running simulation...")
    psi, omega = cavity.solve(verbose=True)

    # Plot results
    cavity.plot_streamfunction(savefig=True)
    cavity.plot_vorticity(savefig=True)

    print("Done!")
