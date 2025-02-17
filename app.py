import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import PCA
from sympy import symbols, expand

# Define weight functions for Z4
hamming_weights = {0: 0, 1: 1, 2: 1, 3: 1}
lee_weights = {0: 0, 1: 1, 2: 2, 3: 1}
euclidean_weights = {0: 0, 1: 1, 2: 4, 3: 1}

# Function to generate codewords
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

# Function to plot PCA with dual code distinction
def plot_pca_with_dual(C, C_dual):
    C_array = np.array(C)
    C_dual_array = np.array(C_dual)
    pca = PCA(n_components=2)
    combined = np.vstack((C_array, C_dual_array))
    transformed = pca.fit_transform(combined)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed[:len(C), 0], transformed[:len(C), 1], color='blue', label='Original Codewords', edgecolor='k', alpha=0.7)
    plt.scatter(transformed[len(C):, 0], transformed[len(C):, 1], color='red', label='Dual Codewords', edgecolor='k', alpha=0.7)
    
    for i, txt in enumerate(combined):
        plt.annotate(f"{list(map(int, txt))}", (transformed[i, 0], transformed[i, 1]), fontsize=10, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.6))
    
    plt.title("PCA Projection of Codewords and Dual Codewords")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid()
    st.pyplot(plt)


# Function to analyze code relationships
def analyze_code_relationship(C, C_dual):
    intersection = sorted(set(C) & set(C_dual))
    is_self_dual = set(C) == set(C_dual)
    is_contained = set(C).issubset(set(C_dual))
    is_self_orthogonal = all(sum(np.array(c) * np.array(d)) % 4 == 0 for c in C for d in C)
    
    st.write("**Code Properties:**")
    st.write("**Intersection of Code and Dual:**")
    st.write("\n".join([f"[{', '.join(map(str, c))}]" for c in intersection]))
    st.write("**Is Self-Dual?**", is_self_dual)
    st.write("**Is Contained in its Dual?**", is_contained)
    st.write("**Is Self-Orthogonal?**", is_self_orthogonal)

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
if st.button("Check if Linear Code"):
    if not is_full_rank(G):
        st.warning("Warning: The generator matrix is not full-rank. This may produce a degenerate code.")
    
    if is_linear_code(C):
        st.success("The code is linear!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Codewords:**")
            st.write("\n".join([f"[{', '.join(map(str, c))}]" for c in C]))
        with col2:
            st.write("**Dual Code:**")
            st.write("\n".join([f"[{', '.join(map(str, c))}]" for c in C_dual]))
        
        plot_pca_with_dual(C, C_dual)
        analyze_code_relationship(C, C_dual)
        
        st.write("**Hamming Distance:**", min_distance(C, hamming_weights))
        st.write("**Lee Distance:**", min_distance(C, lee_weights))
        st.write("**Euclidean Distance:**", min_distance(C, euclidean_weights))
        
        P_H, P_L = compute_weight_enumerators(C)
        st.write("**Hamming Weight Enumerator Polynomial:**", P_H)
        st.write("**Lee Weight Enumerator Polynomial:**", P_L)
    else:
        st.error("The code is not linear.")

