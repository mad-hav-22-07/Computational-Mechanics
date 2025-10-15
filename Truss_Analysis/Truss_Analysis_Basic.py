import numpy as np
import math

# =====================================================================
# ### 1. Input Functions
# =====================================================================

def get_nodal_coordinates():
    """Prompts the user for the total number of nodes and their (x, y) coordinates."""
    while True:
        try:
            num_nodes_str = input("Enter the total number of nodes (e.g., 4): ")
            num_nodes = int(num_nodes_str)
            if num_nodes <= 0:
                print("Number of nodes must be a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print("\n--- Entering Nodal Coordinates (in meters) ---")
    coords = []
    for i in range(num_nodes):
        while True:
            try:
                raw_input = input(f"Enter coordinates for Node {i + 1} (x, y, e.g., 0, 0): ")
                parts = raw_input.replace(',', ' ').split()
                if len(parts) != 2:
                    print("Please enter exactly two values (x and y).")
                    continue
                x = float(parts[0])
                y = float(parts[1])
                coords.append([x, y])
                break
            except ValueError:
                print("Invalid input. Please ensure x and y are valid numbers.")

    return np.array(coords)

def get_member_connectivity():
    """Prompts the user for the total number of members and their start/end node indices."""
    while True:
        try:
            num_members_str = input("Enter the total number of members (elements, e.g., 5): ")
            num_members = int(num_members_str)
            if num_members <= 0:
                print("Number of members must be a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print("\n--- Entering Member Connectivity (1-based indices) ---")
    connectivity = []
    for i in range(num_members):
        while True:
            try:
                raw_input = input(f"Enter start and end nodes for Member {i + 1} (e.g., 1, 3): ")
                parts = raw_input.replace(',', ' ').split()
                if len(parts) != 2:
                    print("Please enter exactly two node indices.")
                    continue
                start_node = int(parts[0])
                end_node = int(parts[1])

                if start_node <= 0 or end_node <= 0:
                    print("Node indices must be positive integers (1-based).")
                    continue

                connectivity.append([start_node, end_node])
                break
            except ValueError:
                print("Invalid input. Please ensure start and end nodes are integers.")

    return np.array(connectivity)

# =====================================================================
# ### 2. Analysis Functions
# =====================================================================

def getElementStiffness(E, A, L, theta):
    """
    Calculates the 4x4 element stiffness matrix (k) in global coordinates
    for a single 2D truss member.
    """
    if L == 0:
        return np.zeros((4, 4))

    # Direction cosines
    c = np.cos(theta)
    s = np.sin(theta)

    # Transformation terms
    c2 = c * c
    s2 = s * s
    cs = c * s

    # Stiffness Matrix Core: k = (AE/L) * ...
    q = np.array([
        [c2, cs, -c2, -cs],
        [cs, s2, -cs, -s2],
        [-c2, -cs, c2, cs],
        [-cs, -s2, cs, s2]
    ])

    # Scale by AE/L
    k = (E * A / L) * q
    return k

def assembleGlobalStiffness(E, A, nodal_coordinates, member_connectivity, num_dofs):
    """Assembles the global stiffness matrix (K) from all element stiffness matrices (k)."""
    K = np.zeros((num_dofs, num_dofs))
    connectivity_0based = member_connectivity - 1

    for i in range(connectivity_0based.shape[0]):
        node1_idx = connectivity_0based[i, 0]
        node2_idx = connectivity_0based[i, 1]

        # Get coordinates
        x1, y1 = nodal_coordinates[node1_idx]
        x2, y2 = nodal_coordinates[node2_idx]

        # Calculate geometric properties
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)
        theta_rad = np.arctan2(dy, dx)

        k_element = getElementStiffness(E, A, L, theta_rad)

        # Global DOFs for this element
        dofs = np.array([
            2 * node1_idx, 2 * node1_idx + 1,
            2 * node2_idx, 2 * node2_idx + 1
        ])

        # Assembly
        for r in range(4):
            for c in range(4):
                K[dofs[r], dofs[c]] += k_element[r, c]

    return K

# =====================================================================
# ### 3. Member Force Function
# =====================================================================

def getMemberForce(E, A, L, theta, u_element):
    """
    Calculates the axial force in a truss member (P) using the element displacement vector.
    """
    # Calculate direction cosines
    c = np.cos(theta)
    s = np.sin(theta)

    # Defines the transformation matrix: T = [-c, -s, c, s]
    T = np.array([-c, -s, c, s])

    # Calculates local deformation (extension/shortening): delta = T @ u_element
    delta = T @ u_element

    # Calculates axial force (P) using P = (A * E / L) * delta
    force = (A * E / L) * delta

    return force

# =====================================================================
# ### 4. Boundary/Load Functions
# =====================================================================

def get_loads(num_dofs):
    """Creates the external global load vector (F) based on user input."""
    F = np.zeros(num_dofs)

    print("\n--- Defining Global Load Vector F (Newtons) ---")

    while True:
        try:
            num_loaded_nodes = int(input("How many nodes have loads applied? "))
            if num_loaded_nodes < 0:
                 print("Please enter a non-negative integer.")
                 continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    for i in range(num_loaded_nodes):
        print(f"\n--- Load {i + 1} of {num_loaded_nodes} ---")
        while True:
            try:
                node_number = int(input("Enter the node number (1-based) where the load is applied: "))

                if node_number <= 0:
                    print("Node number must be positive.")
                    continue

                fx = float(input(f"  Enter force in x-direction (Fx) at Node {node_number} (Newtons): "))
                fy = float(input(f"  Enter force in y-direction (Fy) at Node {node_number} (Newtons): "))

                dof_x = 2 * (node_number - 1)
                dof_y = 2 * (node_number - 1) + 1

                if dof_y >= num_dofs:
                    print(f"Error: Node {node_number} is out of bounds for the defined DOF count ({num_dofs}). Skipping this load.")
                    break

                F[dof_x] += fx
                F[dof_y] += fy
                break

            except ValueError:
                print("Invalid input for node number or force. Please enter integers or numbers.")

    return F

def get_boundary_conditions(num_nodes):
    """Creates a sorted list of constrained Degrees of Freedom (DOFs) based on supports."""
    constrained_dofs = []

    print("\n--- Defining Boundary Conditions (Supports) ---")

    while True:
        try:
            num_supported_nodes = int(input("How many nodes have supports (e.g., pin, roller)? "))
            if num_supported_nodes < 0:
                 print("Please enter a non-negative integer.")
                 continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    for i in range(num_supported_nodes):
        print(f"\n--- Support {i + 1} of {num_supported_nodes} ---")
        while True:
            try:
                node_number = int(input("Enter the node number (1-based) with the support: "))

                if node_number <= 0 or node_number > num_nodes:
                    print(f"Invalid node number. Must be between 1 and {num_nodes}.")
                    continue

                support_type = input("Enter the support type ('pin', 'roller-x', or 'roller-y'): ").lower().strip()

                dof_x = 2 * (node_number - 1)
                dof_y = 2 * (node_number - 1) + 1

                if support_type == 'pin':
                    constrained_dofs.extend([dof_x, dof_y])
                elif support_type == 'roller-x':
                    constrained_dofs.append(dof_x)
                elif support_type == 'roller-y':
                    constrained_dofs.append(dof_y)
                else:
                    print("Invalid support type. Please use 'pin', 'roller-x', or 'roller-y'.")
                    continue

                break

            except ValueError:
                print("Invalid input for node number. Please enter an integer.")

    return np.unique(np.array(constrained_dofs)).astype(int)

# =====================================================================
# --- Main Execution Section ---
# =====================================================================

if __name__ == "__main__":
    print("\n--- 2D Truss Analysis Program ---")

    # 1. Get Geometry from User Input
    nodal_coordinates = get_nodal_coordinates()
    member_connectivity = get_member_connectivity()

    num_nodes = nodal_coordinates.shape[0]
    num_dofs = 2 * num_nodes
    num_members = member_connectivity.shape[0]

    # 2. Define Material and Cross-Sectional Properties
    E = 200e9  # Pascals (200 GPa, typical for steel)
    A = 0.005  # m^2 (50 cm^2)

    # 3. Print Summary Verification
    print("\n" + "="*40)
    print("--- Input Data Summary and Constants ---")
    print("="*40)

    print(f"Total Number of Nodes: {num_nodes}")
    print(f"Total DOFs (Degrees of Freedom): {num_dofs}")
    print(f"Young's Modulus (E): {E:.2e} Pa")
    print(f"Cross-sectional Area (A): {A} m^2")

    print("\nNodal Coordinates (x, y):")
    print(nodal_coordinates)

    print("\nMember Connectivity (Start, End - 1-based index):")
    print(member_connectivity)

    # 4. Member Analysis and Stiffness Calculation (Element Printout)
    print("\n" + "="*60)
    print("--- Calculating Member Properties and Element Stiffness Matrices ---")
    print("="*60)

    member_connectivity_0based = member_connectivity - 1

    # Storage for all member properties needed later (L and theta)
    member_properties = []

    for i in range(num_members):
        node1_1based = member_connectivity[i, 0]
        node2_1based = member_connectivity[i, 1]
        node1_idx = member_connectivity_0based[i, 0]
        node2_idx = member_connectivity_0based[i, 1]
        x1, y1 = nodal_coordinates[node1_idx]
        x2, y2 = nodal_coordinates[node2_idx]
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)
        theta_rad = np.arctan2(dy, dx)
        theta_deg = np.degrees(theta_rad)

        # Store properties for later use
        member_properties.append({
            'L': L,
            'theta': theta_rad,
            'nodes': [node1_idx, node2_idx] # 0-based
        })

        if L > 1e-9:
            k_element = getElementStiffness(E, A, L, theta_rad)
            k_print = k_element / 1e6 # Print in MN/m
        else:
            k_element = np.zeros((4, 4))
            k_print = np.zeros((4, 4))
            print(f"Warning: Member {i + 1} has zero length. Stiffness matrix is zero.")

        # Print Member Summary
        print(f"\nMember {i + 1} (Nodes {node1_1based} to {node2_1based}):")
        print(f"  Length (L): {L:.4f} m")
        print(f"  Angle (θ):  {theta_deg:.4f} degrees ({theta_rad:.4f} radians)")
        print("\n  Element Stiffness Matrix (k / 1e6) [MN/m]:")
        for row in k_print:
            print("  [" + " ".join([f"{val:10.4f}" for val in row]) + " ]")

    print("\nGeometric and element stiffness calculations complete.")

    # 5. Load and Boundary Condition Definition
    F_global = get_loads(num_dofs)
    constrained_dofs = get_boundary_conditions(num_nodes)

    # 6. Assemble Global Stiffness Matrix
    K_global = assembleGlobalStiffness(E, A, nodal_coordinates, member_connectivity, num_dofs)
    print("\nGlobal Stiffness Matrix K assembled.")

    # 7. Solve for Displacements and Reaction Forces
    print("\n" + "="*60)
    print("--- Solving System and Calculating Reactions ---")
    print("="*60)

    # --- Partitioning K and F ---
    all_dofs = np.arange(num_dofs)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_cf = K_global[np.ix_(constrained_dofs, free_dofs)]
    F_f = F_global[free_dofs]
    print(f"System Partitioned: {len(free_dofs)} Free DOFs, {len(constrained_dofs)} Constrained DOFs.")

    # --- Solution ---
    try:
        U_f = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        print("\n!!! ERROR: System is Singular. The truss may be unstable (mechanism). !!!")
        U_f = np.zeros_like(F_f)

    U_global = np.zeros(num_dofs)
    U_global[free_dofs] = U_f
    Reactions = K_cf @ U_f

    # 8. Calculate and Display Member Internal Forces

    # Calculate all forces first
    member_forces = []
    for i, prop in enumerate(member_properties):
        L = prop['L']
        theta_rad = prop['theta']

        node1_idx, node2_idx = prop['nodes']
        dofs = np.array([
            2 * node1_idx, 2 * node1_idx + 1,
            2 * node2_idx, 2 * node2_idx + 1
        ])

        u_element = U_global[dofs]

        if L > 1e-9:
            axial_force = getMemberForce(E, A, L, theta_rad, u_element)
        else:
            axial_force = 0.0

        member_forces.append(axial_force * 1e-3) # Store in kN


    # =====================================================================
    # ### 9. Final Output Formatting (NEW STEP)
    # =====================================================================
    print("\n" + "="*60)
    print("--- FINAL ANALYSIS RESULTS ---")
    print("="*60)

    # 9a. Nodal Displacements Table
    print("\n--- 1. Nodal Displacements (U) ---")
    header_disp = f"{'Node':<4} | {'X-Disp (mm)':>14} | {'Y-Disp (mm)':>14}"
    separator_disp = '-' * len(header_disp)
    print(header_disp)
    print(separator_disp)

    U_print = U_global.reshape(-1, 2)
    for i in range(num_nodes):
        # Convert meters to millimeters (x 1000)
        ux_mm = U_print[i, 0] * 1000
        uy_mm = U_print[i, 1] * 1000
        print(f"{i+1:<4} | {ux_mm:>14.6f} | {uy_mm:>14.6f}")

    # 9b. Reaction Forces Table
    print("\n--- 2. Reaction Forces (R) ---")
    header_react = f"{'Node':<4} | {'Direction':<9} | {'Force (kN)':>12}"
    separator_react = '-' * len(header_react)
    print(header_react)
    print(separator_react)

    for i, dof_index in enumerate(constrained_dofs):
        node = dof_index // 2 + 1
        direction = 'X-dir' if dof_index % 2 == 0 else 'Y-dir'
        force_kN = Reactions[i] * 1e-3

        print(f"{node:<4} | {direction:<9} | {force_kN:>12.3f}")

    # 9c. Member Forces Table
    print("\n--- 3. Member Internal Forces (P) ---")
    header_force = f"{'Member':<6} | {'Nodes':<7} | {'Force (kN)':>12} | {'Status':<15}"
    separator_force = '-' * len(header_force)
    print(header_force)
    print(separator_force)

    for i in range(num_members):
        force_kN = member_forces[i]
        status = "Tension (T)" if force_kN >= 0 else "Compression (C)"

        node1_1based = member_connectivity[i, 0]
        node2_1based = member_connectivity[i, 1]

        print(f"{i+1:<6} | {str(node1_1based) + '-' + str(node2_1based):<7} | {abs(force_kN):>12.3f} | {status:<15}")

    print("\nFinal FEA solution is complete and results are formatted.")

    # Next command for the user
    print("\n--- Next Command ---")
    print("Output 1")