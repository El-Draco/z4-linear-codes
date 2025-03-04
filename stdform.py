
import numpy as np

###############################################################################
# Helper functions for arithmetic in Z4
###############################################################################

def mod4(x):
    """Reduce integer x (or array) modulo 4."""
    return x % 4

def mul_inv_z4(x):
    """
    Return the multiplicative inverse of x in Z4 if it exists,
    or None if x has no inverse.
    The only invertible elements in Z4 are 1 and 3.
    """
    x_mod = x % 4
    if x_mod == 1:
        return 1
    elif x_mod == 3:
        return 3
    else:
        return None

###############################################################################
# Main function: reduce G to standard form (if possible)
###############################################################################

def z4_standard_form(G_in, verbose=True):
    """
    Attempt to bring the generator matrix G_in (over Z4) into
    the standard form:

        [ I_{k1}      A      B1 + 2 B2 ]
        [   0     2 I_{k2}     2 D    ]

    Returns:
        (G_std, k1, k2, success_flag)
    where
        G_std is the transformed matrix,
        k1, k2 are the counts of unit pivots vs. 2-pivots,
        success_flag is True/False indicating if we believe
        we have a valid standard form.
    """
    # Convert input to np.array and reduce mod 4
    G = np.array(G_in, dtype=int)
    G = mod4(G)

    m, n = G.shape

    # We'll keep track of pivot columns in order
    pivot_cols = []
    pivot_types = []  # 1 if pivot is invertible (1 or 3), 2 if pivot is 2

    # row, col pointers for pivot search
    row = 0
    col = 0

    # -------------------------------------------------------------------------
    # First pass: find pivots and do row elimination
    # -------------------------------------------------------------------------
    while row < m and col < n:
        # 1) Look for an invertible pivot (1 or 3) in column col, from row downward
        pivot_row = -1
        pivot_val = 0
        for r in range(row, m):
            val = G[r, col]
            if val in [1, 3]:
                pivot_row = r
                pivot_val = val
                break
        # 2) If none found, look for a pivot of 2
        if pivot_row == -1:
            for r in range(row, m):
                val = G[r, col]
                if val == 2:
                    pivot_row = r
                    pivot_val = val
                    break
        # 3) If still not found, no pivot in this column -> move on
        if pivot_row == -1:
            col += 1
            continue

        # We found a pivot in this column
        # Swap pivot row to the 'row' position
        if pivot_row != row:
            G[[row, pivot_row]] = G[[pivot_row, row]]

        # Normalize pivot if it is invertible
        if pivot_val in [1, 3]:
            inv = mul_inv_z4(pivot_val)  # either 1 or 3
            G[row] = mod4(G[row] * inv)
            pivot_type = 1
        else:
            pivot_type = 2  # pivot_val == 2

        pivot_cols.append(col)
        pivot_types.append(pivot_type)

        # Eliminate below pivot
        if pivot_type == 1:
            # pivot is effectively 1 in G[row, col]
            for r in range(row+1, m):
                if G[r, col] != 0:
                    factor = G[r, col]  # in [1,2,3]
                    G[r] = mod4(G[r] - factor*G[row])
        else:
            # pivot_type == 2
            # we can only eliminate rows that also have a 2 in that column
            for r in range(row+1, m):
                if G[r, col] == 2:
                    # Subtract the pivot row from this row (mod 4)
                    G[r] = mod4(G[r] - G[row])

        row += 1
        col += 1

    # pivot_cols now hold columns that had a pivot
    # pivot_types holds the type (1 or 2) for each pivot in that same order
    k1 = sum(1 for t in pivot_types if t == 1)
    k2 = sum(1 for t in pivot_types if t == 2)

    # -------------------------------------------------------------------------
    # Now reorder the pivot columns so that all type-1 pivots come first,
    # then type-2 pivots, then the non-pivot columns.
    # -------------------------------------------------------------------------
    pivot_cols_1 = []
    pivot_cols_2 = []
    for c, t in zip(pivot_cols, pivot_types):
        if t == 1:
            pivot_cols_1.append(c)
        else:
            pivot_cols_2.append(c)

    pivot_cols_ordered = pivot_cols_1 + pivot_cols_2
    nonpivot_cols = [c for c in range(n) if c not in pivot_cols_ordered]

    new_col_order = pivot_cols_ordered + nonpivot_cols

    # Permute the columns
    G = G[:, new_col_order]

    # -------------------------------------------------------------------------
    # Second pass: we also want to do elimination above the pivots
    # and refine so that the pivot structure is block-diagonal-ish.
    # -------------------------------------------------------------------------
    # We'll do it in two blocks: first the k1 unit pivots, then the k2 2-pivots.
    row = 0
    # Handle the k1 unit pivots (which we want to become an identity block)
    for i in range(k1):
        pivot_col = i
        # The pivot should be in G[row, pivot_col]. We expect it is 1 or 3 now.
        if G[row, pivot_col] not in [1,3]:
            # We fail -> not a valid standard form
            if verbose:
                print("Failed to find an invertible pivot where expected.")
            return (G, k1, k2, False)

        # Normalize pivot to 1
        if G[row, pivot_col] == 3:
            G[row] = mod4(G[row] * 3)  # multiply row by 3 to get pivot = 1

        # Eliminate above pivot
        for r in range(row):
            if G[r, pivot_col] != 0:
                factor = G[r, pivot_col]
                G[r] = mod4(G[r] - factor * G[row])

        # Eliminate below pivot
        for r in range(row+1, m):
            if G[r, pivot_col] != 0:
                factor = G[r, pivot_col]
                G[r] = mod4(G[r] - factor * G[row])

        row += 1

    # Now handle the k2 pivots of type 2 (which we want to become 2 on the diagonal)
    for i2 in range(k2):
        pivot_col = k1 + i2
        # The pivot is at G[row, pivot_col]. We expect it to be 2
        if G[row, pivot_col] != 2:
            # Possibly it is 0, or something else. We try to fix if we can find a row below
            fix_row = -1
            for r in range(row, m):
                if G[r, pivot_col] == 2:
                    fix_row = r
                    break
            if fix_row == -1:
                if verbose:
                    print("Failed to find a '2' pivot where expected.")
                return (G, k1, k2, False)
            # Swap
            if fix_row != row:
                G[[row, fix_row]] = G[[fix_row, row]]

        # Now pivot is 2 in G[row, pivot_col].
        # Clear above
        for r in range(row):
            if G[r, pivot_col] == 2:
                G[r] = mod4(G[r] - G[row])
        # Clear below
        for r in range(row+1, m):
            if G[r, pivot_col] == 2:
                G[r] = mod4(G[r] - G[row])

        row += 1

    # At this point, the top k1 rows should have identity blocks in the first k1 columns,
    # and the next k2 rows should have 2I_{k2} in the next k2 columns.

    # -------------------------------------------------------------------------
    # If we get here, we tentatively say we succeeded
    # -------------------------------------------------------------------------
    if verbose:
        print("Successfully reduced to standard form over Z4.")
        print(f"Found k1={k1} unit pivots, k2={k2} pivots of 2.")
        print("Resulting matrix G_std =\n", G)

    return (G, k1, k2, True)