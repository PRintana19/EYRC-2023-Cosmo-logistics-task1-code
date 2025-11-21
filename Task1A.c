import sympy as sp
import numpy as np
from scipy.linalg import solve_continuous_are

# Define the symbolic variables
x1, x2, u = sp.symbols('x1 x2 u')

# Define the differential equations
x1_dot = -x1 + 2 * x1**3 + x2 + 4 * u
x2_dot = -x1 - x2 + 2 * u

def find_equilibrium_points():
    # Find the equilibrium points by substituting u = 0 and solving the equations.
    # Set u = 0 for equilibrium calculation
    equations = [x1_dot.subs(u, 0), x2_dot.subs(u, 0)]
    # Solve for x1 and x2
    equi_points = sp.solve(equations, (x1, x2))
    return equi_points

def find_A_B_matrices(eq_points):
    # Calculate Jacobian matrices A and B at the equilibrium points.
    # Compute Jacobian matrices for A (state) and B (input)
    A_matrix = sp.Matrix([
        [sp.diff(x1_dot, x1), sp.diff(x1_dot, x2)],
        [sp.diff(x2_dot, x1), sp.diff(x2_dot, x2)]
    ])
    
    B_matrix = sp.Matrix([
        [sp.diff(x1_dot, u)],
        [sp.diff(x2_dot, u)]
    ])
    
    A_matrices, B_matrices = [], []
    
    # Substitute each equilibrium point into A and B matrices
    for point in eq_points:
        A_matrices.append(A_matrix.subs([(x1, point[0]), (x2, point[1])]))
        B_matrices.append(B_matrix.subs([(x1, point[0]), (x2, point[1])]))

    return A_matrices, B_matrices

def find_eigen_values(A_matrices):
    """Find the eigenvalues of all A_matrices and determine system stability."""
    eigen_values = []
    stability = []

    # Calculate eigenvalues for each A matrix
    for A in A_matrices:
        eigenvals = A.eigenvals()  # Get eigenvalues
        eigen_values.append(eigenvals)

        # Check if system is stable (all eigenvalues should have negative real parts)
        if all(ev.as_real_imag()[0] < 0 for ev in eigenvals):
            stability.append("Stable")
        else:
            stability.append("Unstable")

    return eigen_values, stability

def compute_lqr_gain(jacobians_A, jacobians_B):
    """Compute the LQR gain matrix K for unstable equilibrium points."""
    # Define the Q and R matrices
    Q = np.eye(2)  # State weighting matrix, can be tuned for better control
    R = np.array([[1]])  # Control weighting matrix

    K_matrices = []
    
    # Compute LQR only for the unstable points (Point 1 and Point 3)
    for i in [0, 2]:  # Indices for Point 1 and Point 3 (unstable)
        A = np.array(jacobians_A[i]).astype(np.float64)
        B = np.array(jacobians_B[i]).astype(np.float64)

        # Solve the continuous-time algebraic Riccati equation
        P = solve_continuous_are(A, B, Q, R)
        
        # Compute the LQR gain K
        K = np.linalg.inv(R).dot(B.T).dot(P)
        K_matrices.append(K)

    return K_matrices


def main_function():
    """Main function to execute all calculations."""
    # Find equilibrium points
    eq_points = find_equilibrium_points()
    
    if not eq_points:
        print("No equilibrium points found.")
        return None, None, None, None, None, None
    
    # Find Jacobian matrices
    jacobians_A, jacobians_B = find_A_B_matrices(eq_points)
    
    # For finding eigenvalues and stability of the given equation
    eigen_values, stability = find_eigen_values(jacobians_A)
    
    # Compute the LQR gain matrix K for each equilibrium point
    K_matrices = compute_lqr_gain(jacobians_A, jacobians_B)
    
    return eq_points, jacobians_A, eigen_values, stability, K_matrices

def task1a_output(eq_points, jacobians_A, eigen_values, stability, K_matrices):
    """Print the results obtained from calculations."""
    print("Equilibrium Points:")
    for i, point in enumerate(eq_points):
        print(f"  Point {i + 1}: x1 = {point[0]}, x2 = {point[1]}")
    
    print("\nJacobian Matrices at Equilibrium Points:")
    for i, matrix in enumerate(jacobians_A):
        print(f"  At Point {i + 1}:")
        print(sp.pretty(matrix, use_unicode=True))
    
    print("\nEigenvalues at Equilibrium Points:")
    for i, eigvals in enumerate(eigen_values):
        eigvals_str = ', '.join([f"{val}: {count}" for val, count in eigvals.items()])
        print(f"  At Point {i + 1}: {eigvals_str}")
    
    print("\nStability of Equilibrium Points:")
    for i, status in enumerate(stability):
        print(f"  At Point {i + 1}: {status}")
    
    print("\nLQR Gain Matrices K at each Equilibrium Point:")
    for i, K in enumerate(K_matrices):
        print(f"  At Point {i + 1}:")
        print(K)

if __name__ == "__main__":
    results = main_function()
    
    if results:
        eq_points, jacobians_A, eigen_values, stability, K_matrices = results
        task1a_output(eq_points, jacobians_A, eigen_values, stability, K_matrices)
