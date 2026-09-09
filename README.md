# 2D-NS-LidDrivenCavity
# 2D Lid‑Driven Cavity Flow Solver

**A fast, finite‑difference solver for the incompressible Navier–Stokes equations using the vorticity‑streamfunction formulation.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![SciPy](https://img.shields.io/badge/SciPy-Sparse-red)](https://scipy.org/)

---

## 📖 Overview

This repository contains a 2D lid‑driven cavity flow solver written in Python.  
The lid‑driven cavity is a classic benchmark problem in computational fluid dynamics (CFD): a square box with a moving top wall drives a recirculating flow inside the domain.

The solver uses the **vorticity‑streamfunction** formulation, which eliminates pressure from the equations and automatically satisfies the continuity equation. A **sparse direct Poisson solver** (`scipy.sparse.linalg.spsolve`) makes the code fast enough to run on a standard laptop.

---

## 🔬 Governing Equations

We solve the incompressible Navier–Stokes equations in 2D using the vorticity‑streamfunction formulation.

### Vorticity Transport Equation

$$
\frac{\partial \omega}{\partial t} + u \frac{\partial \omega}{\partial x} + v \frac{\partial \omega}{\partial y} = \frac{1}{Re} \left( \frac{\partial^2 \omega}{\partial x^2} + \frac{\partial^2 \omega}{\partial y^2} \right)
$$

### Poisson Equation for Streamfunction

$$
\nabla^2 \psi = -\omega
$$

### Velocity Definitions

$$
u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}
$$

---

## 🧱 Boundary Conditions

| Wall | Streamfunction \( \psi \) | Vorticity \( \omega \) |
|------|--------------------------|------------------------|
| **Top** (moving lid) | \( \psi = 0 \) | \( \omega = -\dfrac{2}{\Delta y^2}\psi_{\text{adj}} + \dfrac{2U_{\text{lid}}}{\Delta y} \) |
| **Bottom** | \( \psi = 0 \) | \( \omega = -\dfrac{2}{\Delta y^2}\psi_{\text{adj}} \) |
| **Left** | \( \psi = 0 \) | \( \omega = -\dfrac{2}{\Delta x^2}\psi_{\text{adj}} \) |
| **Right** | \( \psi = 0 \) | \( \omega = -\dfrac{2}{\Delta x^2}\psi_{\text{adj}} \) |

**Notes:**
- \( \psi_{\text{adj}} \) is the streamfunction value at the grid point adjacent to the wall.
- The top wall condition includes the \( +2U_{\text{lid}}/\Delta y \) term to account for the shear from the moving lid.
- All walls have \( \psi = 0 \) because they are streamlines of the flow.

---

## ⚙️ Numerical Method

| Feature | Implementation |
|---------|----------------|
| **Spatial discretisation** | Central differences (2nd‑order accurate) |
| **Time integration** | Explicit Euler |
| **Poisson solver** | Sparse direct solver via `scipy.sparse.linalg.spsolve` |
| **Stability** | Adaptive time step from CFL and diffusion limits |

### Adaptive Time Step

The time step is computed automatically as:

$$
\Delta t = 0.8 \times \min\left( \frac{0.5 \, \Delta x}{U_{\text{lid}}},\; 0.25 \, Re \, \Delta x^2 \right)
$$

This ensures stability for both the advective and diffusive terms.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Packages: `numpy`, `scipy`, `matplotlib`

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SAPNAMAKHDOOM/2D-NS-LidDrivenCavity.git
cd 2D-NS-LidDrivenCavity
pip install -r requirements.txt
