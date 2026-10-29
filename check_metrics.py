import pickle
with open('models/if_active_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"IF active threshold (offset_): {model.offset_}")