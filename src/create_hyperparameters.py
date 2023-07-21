import pickle
combinations_grid_search = []
basis_size = [10, 30, 50]
learning_rate = [0.01, 0.05, 0.1]
kges = ["TransE", "DistMult", "RotatE", "QuatE"]


for basis in basis_size:
    for rate in learning_rate:
        for kge in kges:
            combinations_grid_search.append((basis, rate, kge))

# Open a file and use dump()
with open('combos.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(combinations_grid_search, file)