import torch

# Extract features from the labeled dataset using the SimCLR encoder
def extract_features(data_loader, encoder):
    all_features = []
    all_labels = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            features = encoder(inputs) 
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_features).to(device), torch.cat(all_labels).to(device)
