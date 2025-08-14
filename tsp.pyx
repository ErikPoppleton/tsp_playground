# tsp_solver.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, exp, log, INFINITY
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.string cimport memcpy
from libc.stdio cimport printf
from scipy.spatial.distance import cdist

np.import_array()

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t INT_t

# C-level distance calculation
cdef inline double euclidean_distance(double* p1, double* p2, int dim) nogil:
    cdef double dist = 0.0
    cdef int i
    for i in range(dim):
        dist += (p1[i] - p2[i]) * (p1[i] - p2[i])
    return sqrt(dist)

# C-level path distance calculation
cdef double calculate_path_distance_c(double[:, :] coords, int[:] path, int n_nodes) nogil:
    cdef double total_dist = 0.0
    cdef int i
    for i in range(n_nodes - 1):
        total_dist += euclidean_distance(&coords[path[i], 0], &coords[path[i+1], 0], 3)
    return total_dist

# Fast path distance calculation using precomputed distance matrix
cdef inline double calculate_path_distance_fast(double[:, :] dist_matrix, int* path, int n_nodes) nogil:
    cdef double total_dist = 0.0
    cdef int i
    for i in range(n_nodes - 1):
        total_dist += dist_matrix[path[i], path[i+1]]
    return total_dist

cdef inline int get_longest_step(double[:, :] dist_matrix, int* path, int n_nodes) nogil:
    cdef double longest = 0.0
    cdef int long_idx
    cdef int i
    cdef double d
    for i in range(n_nodes - 1):
        d = dist_matrix[path[i], path[i+1]]
        if d > longest:
            longest = d
            long_idx = i
    return long_idx

cdef inline double get_length_from(double[:, :] dist_matrix, int* path, int i) nogil:
    return dist_matrix[path[i], path[i+1]]

# Better initial heuristic
cdef int lagging_neighbor_path_c(double[:, :] coords, int start_index, int* path, int n_nodes) nogil:
    cdef int i, j, current, nearest
    cdef double min_dist, dist
    cdef double* lagging_mean = <double*>malloc(3 * sizeof(double))
    cdef double* ref = <double*>malloc(3 * sizeof(double))
    cdef bint* visited = <bint*>malloc(n_nodes * sizeof(bint))

    # Initialize visited array
    for i in range(n_nodes):
        visited[i] = False
    
    # Start from the given node
    path[0] = start_index
    visited[start_index] = True
    current = start_index
    for i in range(3):
        lagging_mean[i] = coords[start_index][i]

    # Build path less greedily
    for i in range(1, n_nodes):
        min_dist = INFINITY
        nearest = -1
        for j in range(3):
            ref[j] = lagging_mean[j] / min(10, i)

        for j in range(n_nodes):
            if not visited[j]:
                dist = euclidean_distance(&coords[j, 0], ref, 3)
                if dist < min_dist:
                    min_dist = dist
                    nearest = j

        if nearest != -1:
            path[i] = nearest
            visited[nearest] = True
            current = nearest
            # Update the lagging mean
            for j in range(3):
                lagging_mean[j] += coords[nearest][j]
                if i > 10:
                    lagging_mean[j] -= coords[i-10][j]

    
    free(visited)
    return 0

def lagging_neighbor_path_py(np.ndarray[DTYPE_t, ndim=2] node_coords, int start_index):
    cdef int n_nodes = len(node_coords)
    cdef int i
    cdef double* p1
    cdef double* p2
    
    # Validate start_index
    if start_index < 0 or start_index >= n_nodes:
        raise ValueError(f"start_index {start_index} is out of bounds for {n_nodes} nodes")
    
    # Convert to contiguous array if needed
    if not node_coords.flags['C_CONTIGUOUS']:
        node_coords = np.ascontiguousarray(node_coords)
    
    # Get initial solution using nearest neighbor heuristic
    cdef np.ndarray[INT_t, ndim=1] nn_path = np.zeros(n_nodes, dtype=np.int32)
    cdef int* nn_path_ptr = <int*>nn_path.data

    lagging_neighbor_path_c(node_coords, start_index, nn_path_ptr, n_nodes)

    # Check quality of nearest neighbor solution
    cdef np.ndarray[DTYPE_t, ndim=1] edge_distances = np.zeros(n_nodes - 1, dtype=np.float64)
    cdef double* node_coords_ptr = <double*>node_coords.data
    cdef int n_dims = node_coords.shape[1]  # Get number of dimensions
    
    for i in range(n_nodes - 1):
        # Get pointers to the specific rows in the 2D array
        p1 = node_coords_ptr + nn_path[i] * n_dims
        p2 = node_coords_ptr + nn_path[i+1] * n_dims
        edge_distances[i] = euclidean_distance(p1, p2, n_dims)
    
    cdef double nn_distance = np.sum(edge_distances)
    cdef double min_dist = np.min(edge_distances)
    cdef double max_dist = np.max(edge_distances)
    cdef double avg_dist = np.mean(edge_distances)
    print(f"Nearest neighbor path distance: {nn_distance:.4f}")
    print(f"Edge statistics - Min: {min_dist:.4f}, Max: {max_dist:.4f}, Avg: {avg_dist:.4f}")
    return nn_path

# Optimized nearest neighbor heuristic
cdef int nearest_neighbor_tsp_path_c(double[:, :] dist_matrix, int start_index, int* path, int n_nodes) nogil:
    cdef int i, j, current, nearest
    cdef double min_dist, dist
    cdef bint* visited = <bint*>malloc(n_nodes * sizeof(bint))
    
    # Initialize visited array
    for i in range(n_nodes):
        visited[i] = False
    
    # Start from the given node
    path[0] = start_index
    visited[start_index] = True
    current = start_index
    
    # Build path greedily
    for i in range(1, n_nodes):
        min_dist = INFINITY
        nearest = -1
        
        for j in range(n_nodes):
            if not visited[j]:
                dist = dist_matrix[current, j]
                if dist < min_dist:
                    min_dist = dist
                    nearest = j
        
        if nearest != -1:
            path[i] = nearest
            visited[nearest] = True
            current = nearest
    
    free(visited)
    return 0

# 2-opt swap for local optimization
cdef inline double two_opt_swap_delta(double[:, :] dist_matrix, int* path, int i, int j, int n_nodes) nogil:
    """Calculate the change in distance from a 2-opt swap between positions i and j."""
    
    # Current edges: (i-1, i) and (j, j+1)
    # After reversal: (i-1, j) and (i, j+1)
    # i cannot be 0
    cdef double current_dist = dist_matrix[path[i-1], path[i]]
    cdef double new_dist = dist_matrix[path[i-1], path[j]]

    # j can be the last node 
    if j != n_nodes - 1:
        current_dist += dist_matrix[path[j], path[j+1]]
        new_dist += dist_matrix[path[i], path[j+1]]
    
    return new_dist - current_dist

# Perform 2-opt swap
cdef int perform_2opt_swap(int* path, int i, int j) nogil:
    """Reverse the path segment between i and j (inclusive)."""
    cdef int left = i
    cdef int right = j
    cdef int temp
    
    while left < right:
        temp = path[left]
        path[left] = path[right]
        path[right] = temp
        left += 1
        right -= 1

    return 0

# 3-opt move evaluation and execution
cdef double evaluate_3opt_move(double[:, :] dist_matrix, int* path, int i, int j, int k, int n_nodes, int* best_case) nogil:
    """Evaluate all possible 3-opt reconnections and return the best improvement."""
    if i <= 0 or k >= n_nodes - 1 or j <= i + 1 or k <= j + 1:
        return 0.0
        
    cdef double current_cost, new_cost, best_delta = 0.0
    
    # Current cost of the three edges
    current_cost = (dist_matrix[path[i-1], path[i]] + 
                   dist_matrix[path[j-1], path[j]] + 
                   dist_matrix[path[k-1], path[k]])
    
    # Case 1: Reverse segment (i, j-1)
    new_cost = (dist_matrix[path[i-1], path[j-1]] + 
                dist_matrix[path[i], path[j]] + 
                dist_matrix[path[k-1], path[k]])
    if new_cost - current_cost < best_delta:
        best_delta = new_cost - current_cost
        best_case[0] = 1
    
    # Case 2: Reverse segment (j, k-1)
    new_cost = (dist_matrix[path[i-1], path[i]] + 
                dist_matrix[path[j-1], path[k-1]] + 
                dist_matrix[path[j], path[k]])
    if new_cost - current_cost < best_delta:
        best_delta = new_cost - current_cost
        best_case[0] = 2
    
    # Case 3: Reverse both segments
    new_cost = (dist_matrix[path[i-1], path[j-1]] + 
                dist_matrix[path[i], path[k-1]] + 
                dist_matrix[path[j], path[k]])
    if new_cost - current_cost < best_delta:
        best_delta = new_cost - current_cost
        best_case[0] = 3
    
    # Case 4: Rotate segments
    new_cost = (dist_matrix[path[i-1], path[j]] + 
                dist_matrix[path[k-1], path[i]] + 
                dist_matrix[path[j-1], path[k]])
    if new_cost - current_cost < best_delta:
        best_delta = new_cost - current_cost
        best_case[0] = 4
    
    return best_delta

# Random number generator (0 to 1)
cdef inline double rand_uniform() nogil:
    return <double>rand() / <double>RAND_MAX

# Simulated Annealing solver
cdef class SimulatedAnnealing:
    cdef double[:, :] dist_matrix
    cdef int n_nodes
    cdef int* current_path
    cdef int* best_path
    cdef int* temp_path  # For temporary operations
    cdef double current_cost
    cdef double best_cost
    cdef double temperature
    cdef double cooling_rate
    cdef int max_iterations
    cdef int max_iterations_without_improvement
    
    def __init__(self, double[:, :] dist_matrix, int n_nodes):
        self.dist_matrix = dist_matrix
        self.n_nodes = n_nodes
        self.current_path = <int*>malloc(n_nodes * sizeof(int))
        self.best_path = <int*>malloc(n_nodes * sizeof(int))
        self.temp_path = <int*>malloc(n_nodes * sizeof(int))
        self.temperature = 100.0
        self.cooling_rate = 0.995
        self.max_iterations = 500000
        self.max_iterations_without_improvement = 1000000
        
    def __dealloc__(self):
        if self.current_path != NULL:
            free(self.current_path)
        if self.best_path != NULL:
            free(self.best_path)
        if self.temp_path != NULL:
            free(self.temp_path)
    
    cdef void initialize(self, int* initial_path):
        """Initialize with a given path."""
        cdef int i
        for i in range(self.n_nodes):
            self.current_path[i] = initial_path[i]
            self.best_path[i] = initial_path[i]
        
        self.current_cost = calculate_path_distance_fast(self.dist_matrix, self.current_path, self.n_nodes)
        self.best_cost = self.current_cost
        
        # Set initial temperature based on average edge length
        cdef double avg_dist = self.current_cost / (self.n_nodes - 1)
        self.temperature = avg_dist * 2.0  # Start with temperature proportional to problem scale
    
    cdef bint accept_move(self, double delta) nogil:
        """Decide whether to accept a move based on the Metropolis criterion."""
        if delta < 0:  # Improvement
            return True
        elif self.temperature > 1e-10:
            # Accept worse solution with probability exp(-delta/temperature)
            return rand_uniform() < exp(-delta / self.temperature)
        return False
    
    cdef void optimize(self):
        """Run simulated annealing optimization."""
        cdef int iteration = 0
        cdef int iterations_without_improvement = 0
        cdef int i, j, k, move_type
        cdef double delta
        cdef int best_case
        cdef int temp_node
        cdef int longest_step
        cdef double longest_length
        
        print(f"Starting simulated annealing with initial cost: {self.current_cost:.4f}")
        
        while iteration < self.max_iterations:
            # Choose move type based on temperature (more complex moves at higher temperatures)
            r = rand_uniform()
            if self.temperature > 10 and r < 0.3:
                move_type = 3  # 3-opt
            elif r < 0.7:
                move_type = 2  # 2-opt
            else:
                move_type = 1  # Simple swap
            
            delta = 0.0

            if move_type == 1:
                # Simple swap: exchange two random nodes (except the first one)
                # Every 100 moves force an action on the longest jump
                if iteration % 200 - 100 == 0:
                    i = get_longest_step(self.dist_matrix, self.current_path, self.n_nodes)
                elif iteration % 200 == 0:
                    i = get_longest_step(self.dist_matrix, self.current_path, self.n_nodes)
                    i += 1
                else:
                    i = 1 + <int>(rand_uniform() * (self.n_nodes - 1))

                j = 1 + <int>(rand_uniform() * (self.n_nodes - 1))
                
                # Ensure we get different indices
                while j == i:
                    j = 1 + <int>(rand_uniform() * (self.n_nodes - 1))
                
                # Calculate delta
                delta = self.calculate_swap_delta(i, j)
                
                if self.accept_move(delta):
                    # Perform swap
                    temp_node = self.current_path[i]
                    self.current_path[i] = self.current_path[j]
                    self.current_path[j] = temp_node
                    self.current_cost += delta
                    
            elif move_type == 2:
                # 2-opt: reverse a segment
                # Select two positions ensuring valid range
                if iteration % 200 - 100 == 0:
                    longest_step = get_longest_step(self.dist_matrix, self.current_path, self.n_nodes)
                    if longest_step > 0 and longest_step < self.n_nodes - 2:
                        i = longest_step
                    else:
                        i = 1 + <int>(rand_uniform() * (self.n_nodes - 3))
                else:
                    i = 1 + <int>(rand_uniform() * (self.n_nodes - 3))
                if iteration % 200 == 0:
                    longest_step = get_longest_step(self.dist_matrix, self.current_path, self.n_nodes)
                    longest_step += 1 # ls is the index of the start of the step, get the end.
                    if longest_step < self.n_nodes - 1:
                        j = longest_step
                        i = max(1, j - 19) + <int>(rand_uniform() * min(j, 19))
                    else:
                        j = j = i + 2 + <int>(rand_uniform() * min(self.n_nodes - i - 2, 20))  # Limit segment size
                else:        
                    j = i + 2 + <int>(rand_uniform() * min(self.n_nodes - i - 2, 20))  # Limit segment size
                
                delta = two_opt_swap_delta(self.dist_matrix, self.current_path, i, j, self.n_nodes)
                
                if self.accept_move(delta):
                    perform_2opt_swap(self.current_path, i, j)
                    self.current_cost += delta
                    
            else:  # 3-opt
                # Select three edges to remove with valid ranges
                i = 1 + <int>(rand_uniform() * max(1, (self.n_nodes - 6) // 3))
                j = i + 2 + <int>(rand_uniform() * max(1, (self.n_nodes - i - 4) // 2))
                k = j + 2 + <int>(rand_uniform() * max(1, min(self.n_nodes - j - 2, 10)))
                
                if k >= self.n_nodes:
                    k = self.n_nodes - 1
                
                best_case = 0
                delta = evaluate_3opt_move(self.dist_matrix, self.current_path, i, j, k, 
                                          self.n_nodes, &best_case)
                
                if best_case > 0 and self.accept_move(delta):
                    # Apply the best 3-opt reconnection
                    self.apply_3opt_move(i, j, k, best_case)
                    self.current_cost += delta

            if self.current_cost < 0:
                print("ERROR!")
                print(f"Current cost: {self.current_cost}, Last delta: {delta}, move type: {move_type}, Affected nodes: {i}, {j}")
            
            # Sanity check - recalculate cost occasionally to avoid drift
            if iteration % 5000 == 0 and iteration > 0:
                actual_cost = calculate_path_distance_fast(self.dist_matrix, self.current_path, self.n_nodes)
                if abs(actual_cost - self.current_cost) > 1e-6:
                    print(f"Cost drift detected at iteration {iteration}: tracked={self.current_cost:.6f}, actual={actual_cost:.6f}")
                    self.current_cost = actual_cost
            
            # Update best solution if improved
            if self.current_cost < self.best_cost - 1e-9:  # Small tolerance for floating point
                self.best_cost = self.current_cost
                memcpy(self.best_path, self.current_path, self.n_nodes * sizeof(int))
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            
            # Cool down
            self.temperature *= self.cooling_rate
            
            # Periodic progress report
            if iteration % 10000 == 0:
                longest_step = get_longest_step(self.dist_matrix, self.current_path, self.n_nodes)
                longest_length = get_length_from(self.dist_matrix, self.current_path, longest_step)
                print(f"Iteration {iteration}: Current cost = {self.current_cost:.4f}, "
                      f"Best cost = {self.best_cost:.4f}, Temperature = {self.temperature:.6f}, "
                      f"Longest step = {longest_length} at position {longest_step}")
            
            # Reheat if stuck in local optimum
            if iterations_without_improvement > 2000 and self.temperature < 1.0:
                self.temperature = 10.0  # Moderate reheat
                #print(f"Reheating at iteration {iteration}")

            if iterations_without_improvement >= self.max_iterations_without_improvement:
                print(f"No improvement in {iterations_without_improvement} iterations, resetting to best path")
                self.current_cost = self.best_cost
                memcpy(self.current_path, self.best_path, self.n_nodes * sizeof(int))
                self.temperature = 1
                iterations_without_improvement = 0
            
            iteration += 1
        
        print(f"Optimization complete after {iteration} iterations. Final best cost: {self.best_cost:.4f}")
    
    cdef double calculate_swap_delta(self, int i, int j) nogil:
        """Calculate the change in total distance from swapping nodes at positions i and j."""
        if i == j:
            return 0.0
            
        cdef double delta = 0.0
        cdef int node_i = self.current_path[i]
        cdef int node_j = self.current_path[j]
        
        # Make sure i < j for consistency
        cdef int temp
        if i > j:
            temp = i
            i = j
            j = temp
            temp = node_i
            node_i = node_j
            node_j = temp
        
        # Special case: adjacent nodes
        if j == i + 1:
            # Remove old edges
            if i > 0:
                delta -= self.dist_matrix[self.current_path[i-1], node_i]
                delta += self.dist_matrix[self.current_path[i-1], node_j]
            if j < self.n_nodes - 1:
                delta -= self.dist_matrix[node_j, self.current_path[j+1]]
                delta += self.dist_matrix[node_i, self.current_path[j+1]]
        else:
            # Non-adjacent nodes
            # Edges connected to position i
            if i > 0:
                delta -= self.dist_matrix[self.current_path[i-1], node_i]
                delta += self.dist_matrix[self.current_path[i-1], node_j]
            if i < self.n_nodes - 1:
                delta -= self.dist_matrix[node_i, self.current_path[i+1]]
                delta += self.dist_matrix[node_j, self.current_path[i+1]]
            
            # Edges connected to position j
            if j > 0:
                delta -= self.dist_matrix[self.current_path[j-1], node_j]
                delta += self.dist_matrix[self.current_path[j-1], node_i]
            if j < self.n_nodes - 1:
                delta -= self.dist_matrix[node_j, self.current_path[j+1]]
                delta += self.dist_matrix[node_i, self.current_path[j+1]]
        
        return delta
    
    cdef void apply_3opt_move(self, int i, int j, int k, int case_num):
        """Apply the specified 3-opt reconnection."""
        cdef int pos = 0, idx
        
        # Copy to temp path based on case
        if case_num == 1:  # Reverse segment (i, j-1)
            for idx in range(i):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(j-1, i-1, -1):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(j, self.n_nodes):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
                
        elif case_num == 2:  # Reverse segment (j, k-1)
            for idx in range(j):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(k-1, j-1, -1):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(k, self.n_nodes):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
                
        elif case_num == 3:  # Reverse both segments
            for idx in range(i):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(j-1, i-1, -1):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(k-1, j-1, -1):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(k, self.n_nodes):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
                
        elif case_num == 4:  # Rotate segments
            for idx in range(i):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(j, k):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(i, j):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
            for idx in range(k, self.n_nodes):
                self.temp_path[pos] = self.current_path[idx]
                pos += 1
        
        # Copy back from temp to current
        memcpy(self.current_path, self.temp_path, self.n_nodes * sizeof(int))
    
    def get_best_path(self):
        """Return the best path found."""
        cdef np.ndarray[INT_t, ndim=1] result = np.zeros(self.n_nodes, dtype=np.int32)
        cdef int i
        for i in range(self.n_nodes):
            result[i] = self.best_path[i]
        return result

# Main TSP solver function
def tsp_path_solver(np.ndarray[DTYPE_t, ndim=2] node_coords, int start_index, 
                   double cooling_rate=0.995, int max_iterations=1000000):
    """
    Solve the Traveling Salesperson Path problem using simulated annealing.
    
    Args:
        node_coords: (n_nodes, 3) numpy array containing the (x, y, z) positions of the nodes
        start_index: Index of the starting node
        cooling_rate: Temperature cooling rate (default 0.995)
        max_iterations: Maximum number of iterations (default 100000)
        
    Returns:
        numpy.ndarray: Array of node indices representing the optimized path order
    """
    cdef int n_nodes = len(node_coords)
    cdef int i
    
    # Validate start_index
    if start_index < 0 or start_index >= n_nodes:
        raise ValueError(f"start_index {start_index} is out of bounds for {n_nodes} nodes")
    
    # Convert to contiguous array if needed
    if not node_coords.flags['C_CONTIGUOUS']:
        node_coords = np.ascontiguousarray(node_coords)
    
    # Precompute distance matrix
    cdef np.ndarray[DTYPE_t, ndim=2] dist_matrix = cdist(node_coords, node_coords)
    
    # Get initial solution using nearest neighbor heuristic
    cdef np.ndarray[INT_t, ndim=1] nn_path = np.zeros(n_nodes, dtype=np.int32)
    cdef int* nn_path_ptr = <int*>nn_path.data
    
    print(f"Solving TSP for {n_nodes} nodes starting from index {start_index}")
    print("Step 1: Finding greedy nearest neighbor solution...")
    
    #nearest_neighbor_tsp_path_c(dist_matrix, start_index, nn_path_ptr, n_nodes)
    lagging_neighbor_path_c(node_coords, start_index, nn_path_ptr, n_nodes)

    # Calculate nearest neighbor distance
    cdef double nn_distance = calculate_path_distance_fast(dist_matrix, nn_path_ptr, n_nodes)
    
    # Check quality of nearest neighbor solution
    cdef np.ndarray[DTYPE_t, ndim=1] edge_distances = np.zeros(n_nodes - 1, dtype=np.float64)
    for i in range(n_nodes - 1):
        edge_distances[i] = dist_matrix[nn_path[i], nn_path[i+1]]
    
    cdef double min_dist = np.min(edge_distances)
    cdef double max_dist = np.max(edge_distances)
    cdef double avg_dist = np.mean(edge_distances)
    
    print(f"Nearest neighbor path distance: {nn_distance:.4f}")
    print(f"Edge statistics - Min: {min_dist:.4f}, Max: {max_dist:.4f}, Avg: {avg_dist:.4f}")
    
    # Always run simulated annealing for problems of this size
    print("\nStep 2: Running simulated annealing optimization...")
    
    # Initialize simulated annealing solver
    cdef SimulatedAnnealing solver = SimulatedAnnealing(dist_matrix, n_nodes)
    solver.cooling_rate = cooling_rate
    solver.max_iterations = max_iterations
    
    # Start from nearest neighbor solution
    solver.initialize(nn_path_ptr)
    
    # Run optimization
    solver.optimize()
    
    # Get the best path found
    cdef np.ndarray[INT_t, ndim=1] best_path = solver.get_best_path()
    
    # Calculate final statistics
    cdef double final_distance = calculate_path_distance_fast(dist_matrix, <int*>best_path.data, n_nodes)
    cdef double improvement = (nn_distance - final_distance) / nn_distance * 100
    
    print(f"\nOptimization Results:")
    print(f"Initial distance: {nn_distance:.4f}")
    print(f"Final distance: {final_distance:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    return best_path

# Python wrapper for calculate_path_distance
def calculate_path_distance(np.ndarray[DTYPE_t, ndim=2] coords, np.ndarray[INT_t, ndim=1] path):
    """Calculate total distance for a given path."""
    return calculate_path_distance_c(coords, path, len(path))