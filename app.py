import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import PCA
from sympy import symbols, expand
from stdform import z4_standard_form

# Define weight functions for Z4
hamming_weights = {0: 0, 1: 1, 2: 1, 3: 1}
lee_weights = {0: 0, 1: 1, 2: 2, 3: 1}
euclidean_weights = {0: 0, 1: 1, 2: 4, 3: 1}

def min_hamming_distance_binary(C_bin):
    def hamming_dist(u, v):
        return sum(ui != vi for ui, vi in zip(u, v))
    distances = [hamming_dist(u, v) for u, v in product(C_bin, repeat=2) if u != v]
    return min(distances) if distances else 0


# Function to apply the Gray map to a codeword
def gray_map(codeword):
    gray_mapping = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    return [bit for num in codeword for bit in gray_mapping[num]]

# Function to generate codewords and their Gray-mapped versions
def generate_gray_mapped_codewords(G):
    codewords = generate_codewords(G)
    gray_codewords = [gray_map(c) for c in codewords]
    return codewords, gray_codewords

def generate_codewords(G):
    k, n = G.shape
    codewords = set()
    for coeffs in product(range(4), repeat=k):
        codeword = tuple(np.mod(np.dot(coeffs, G), 4))
        codewords.add(codeword)
    return sorted(codewords)

# Function to check linearity
def is_linear_code(C):
    for u, v in product(C, repeat=2):
        sum_uv = tuple(np.mod(np.array(u) + np.array(v), 4))
        if sum_uv not in C:
            return False
    return True

# Function to check full-rank
def is_full_rank(G):
    return np.linalg.matrix_rank(G) == G.shape[0]

# Function to compute minimum distance
def min_distance(C, weight_fn):
    distances = [sum(weight_fn[x] for x in np.mod(np.array(u) - np.array(v), 4))
                 for u, v in product(C, repeat=2) if u != v]
    return min(distances) if distances else 0

# Function to compute the dual code
def dual_code(G):
    k, n = G.shape
    all_vectors = list(product(range(4), repeat=n))
    dual_basis = [v for v in all_vectors if all(np.dot(v, row) % 4 == 0 for row in G)]
    return dual_basis


# Function to analyze code relationships
def analyze_code_relationship(C, C_dual):
    intersection = sorted(set(C) & set(C_dual))
    is_self_dual = set(C) == set(C_dual)
    is_contained = set(C).issubset(set(C_dual))
    is_self_orthogonal = all(sum(np.array(c) * np.array(d)) % 4 == 0 for c in C for d in C)
    is_lcd = intersection == [(0,) * len(C[0])]  # LCD check: Only zero vector in intersection

    st.write("## Code Properties")
    st.write("### Intersection of Code and Dual:")
    st.write("\n".join([f"[{', '.join(map(str, c))}]" for c in intersection]) or "[0, 0, ..., 0]")

    st.write("**Is Self-Dual?**", is_self_dual)
    st.write("**Is Contained in its Dual?**", is_contained)
    st.write("**Is Self-Orthogonal?**", is_self_orthogonal)
    st.write("**Is LCD Code?**", is_lcd)


# Function to compute weight enumerators
def compute_weight_enumerators(C):
    X, Y = symbols('X Y')
    P_H = sum(X**(len(C[0]) - sum(hamming_weights[x] for x in c)) * Y**sum(hamming_weights[x] for x in c) for c in C)
    P_L = sum(X**(2*len(C[0]) - sum(lee_weights[x] for x in c)) * Y**sum(lee_weights[x] for x in c) for c in C)
    return expand(P_H), expand(P_L)

# Streamlit UI
st.title("Z4 Code Explorer")

# User Input: Rows and Columns
k = st.number_input("Number of Rows (k)", min_value=1, max_value=10, value=2, step=1)
n = st.number_input("Number of Columns (n)", min_value=2, max_value=10, value=4, step=1)

# Matrix Input or Random Generation
if "G" not in st.session_state:
    st.session_state.G = np.random.randint(0, 4, (k, n))

matrix_option = st.radio("Matrix Input Mode", ["Random", "Manual"], index=0)

if matrix_option == "Random":
    if st.button("Generate New Random Matrix"):
        st.session_state.G = np.random.randint(0, 4, (k, n))
    G = st.session_state.G
else:
    G = np.zeros((k, n), dtype=int)
    for i in range(k):
        user_input = st.text_input(f"Row {i+1} (comma-separated)", value='')
        if user_input:
            G[i] = np.array(user_input.split(','), dtype=int)

st.write("**Generator Matrix:**")
st.write(G)

# Compute Codewords
C = generate_codewords(G)
C_dual = dual_code(G)

if st.button("Compute Metrics"):
    if not is_full_rank(G):
        st.warning("Warning: The generator matrix is not full-rank. This may produce a degenerate code.")

    if is_linear_code(C):
        st.success("The code is linear!")

        # Compute standard form
        G_std, k1, k2, success = z4_standard_form(G, verbose=False)

        # Compute the number of codewords
        num_codewords = len(C)
        num_dual_codewords = len(C_dual)

        # Row 1: Show Codewords & Dual Codewords Side by Side
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### Codewords ({num_codewords} total)")
            st.write("\n".join([f"[{', '.join(map(str, c))}]" for c in C]))
        with col2:
            st.write(f"### Dual Codewords ({num_dual_codewords} total)")
            st.write("\n".join([f"[{', '.join(map(str, c))}]" for c in C_dual]))

        C, C_gray = generate_gray_mapped_codewords(G)
        st.write(r"### Gray-Mapped Codewords:")
        st.write("\n".join([f"[{', '.join(map(str, c))}]" for c in C_gray]))

        # Count codewords
        st.write(f"### Number of Gray-Mapped Codewords: {len(C_gray)}")

        # Compute minimum Hamming distance for Gray image
        gray_hamming_dist = min_hamming_distance_binary(C_gray)

        st.metric(label="Hamming Distance (Gray Image)", value=gray_hamming_dist)


        # Row 1: Show Generator Matrix & Standard Form Matrix
        col3, col4 = st.columns(2)
        with col3:
            st.write("### Generator Matrix")
            st.write(G)
        with col4:
            if success:
                st.write("### Standard Form Matrix")
                st.write(G_std)
            else:
                st.write("### Standard Form Matrix")
                st.error("Conversion to standard form failed.")

        # Row 2: Print the Standard Form Format
        st.write("### Expected Standard Form Structure:")
        st.latex(r"""
        \begin{bmatrix}
        I_{k1} & A & B_1 + 2 B_2 \\
        0 & 2 I_{k2} & 2 D
        \end{bmatrix}
        """)

        # Row 3: Display k1 and k2
        st.write(f"#### **k1=** {k1}, **k2=** {k2}")

        if success:
            # Extract matrix components
            A = G_std[:k1, k1:k1 + (n - (k1 + k2))]
            B1 = G_std[:k1, k1 + (n - (k1 + k2)):n-k2] // 2
            B2 = (G_std[:k1, k1 + (n - (k1 + k2)):n-k2] % 2) // 1
            D = G_std[k1:, n-k2:] // 2

            # Row 4: Show A, B1, B2, D side by side
            col5, col6 = st.columns(2)
            with col5:
                st.write("### A Matrix")
                st.write(A)
                st.write("### D Matrix")
                st.write(D)
            with col6:
                st.write("### B1 Matrix")
                st.write(B1)
                st.write("### B2 Matrix")
                st.write(B2)
        else:
            st.error("Standard form conversion failed, submatrices cannot be extracted.")

        # Analyze relationships
        analyze_code_relationship(C, C_dual)

        # Display distances
        st.write("## Distance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Hamming Distance", value=min_distance(C, hamming_weights))
        with col2:
            st.metric(label="Lee Distance", value=min_distance(C, lee_weights))
        with col3:
            st.metric(label="Euclidean Distance", value=min_distance(C, euclidean_weights))

        # Compute weight enumerators
        st.write("## Weight Enumerators")
        P_H, P_L = compute_weight_enumerators(C)

        st.write("### Hamming Weight Enumerator Polynomial:")
        st.latex(P_H)

        st.write("### Lee Weight Enumerator Polynomial:")
        st.latex(P_L)
    else:
        st.error("The code is not linear.")
