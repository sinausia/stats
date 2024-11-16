import numpy as np
import pandas as pd

scores_path = '...scores.csv'
scores_df = pd.read_csv(scores_path)


scores_df = scores_df.iloc[:910, :15]
scores = scores_df.values
def create_hankel_matrix(data, hankel_depth):
    rows, cols = data.shape
    hankel_matrix = []
    for i in range(hankel_depth):
        hankel_matrix.append(data[i:rows-hankel_depth+i+1, :])
    return np.hstack(hankel_matrix)


def create_hankel_prime_matrix(data, hankel_depth):
    rows, cols = data.shape
    hankel_matrix = []
    for i in range(hankel_depth):
        hankel_matrix.append(data[i:rows-hankel_depth+i+1, i+1:])
    return np.hstack(hankel_matrix)

# Define the depth of the Hankel matrix (number of shifts)
hankel_depth = 10

# Create the Hankel matrix
hankel_matrix = create_hankel_matrix(scores, hankel_depth)
hankel_matrix_prime = create_hankel_prime_matrix(scores, hankel_depth)

hankel_df = pd.DataFrame(hankel_matrix)
hankel_prime_df = pd.DataFrame(hankel_matrix_prime)

hankel_path = '/Users/danielsinausia/Documents/test/Hankel_matrix.csv'
hankel_df.to_csv(hankel_path, index=False)

hankel_prime_path = '/Users/danielsinausia/Documents/test/Hankel_matrix_prime.csv'
hankel_prime_df.to_csv(hankel_path, index=False)


#%% SVD of the Hankel matrix

U, S, Vt = np.linalg.svd(hankel_matrix, full_matrices=False)

# Save the singular values to a CSV file
singular_values_df = pd.DataFrame(S, columns=['Singular Values'])
singular_values_path = '/Users/danielsinausia/Documents/test/Singular_values.csv'
singular_values_df.to_csv(singular_values_path, index=False)



#%% A, B, C, D
U, S, Vt = np.linalg.svd(hankel_matrix, full_matrices=False)

# Determine the order of the system (number of significant singular values)
threshold = 1e-5
r = np.sum(S > threshold)
print(f"System order (r): {r}")

# Truncate U, S, Vt based on the system order r
U_r = U[:, :r]
S_r = np.diag(S[:r])
Vt_r = Vt[:r, :]



A = np.linalg.inv(np.sqrt(S_r)) @ U_r.T @ hankel_matrix_prime @ Vt_r.T @ np.linalg.inv(np.sqrt(S_r))
B = H_f[:, r:]
C = H_f[:r, :]
D = scores.iloc[0, :]

A_df = pd.DataFrame(A)
B_df = pd.DataFrame(B)
C_df = pd.DataFrame(C)
D_df = pd.DataFrame(D)

A_path = '/Users/danielsinausia/Documents/test/Matrix_A.csv'
B_path = '/Users/danielsinausia/Documents/test/Matrix_B.csv'
C_path = '/Users/danielsinausia/Documents/test/Matrix_C.csv'
D_path = '/Users/danielsinausia/Documents/test/Matrix_D.csv'

A_df.to_csv(A_path, index=False)
B_df.to_csv(B_path, index=False)
C_df.to_csv(C_path, index=False)
D_df.to_csv(D_path, index=False)

