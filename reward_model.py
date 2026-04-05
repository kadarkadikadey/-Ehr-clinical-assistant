import torch
import torch.nn as nn

class RewardPredictor(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def extract_features(obs_data):
    """
    Safely converts Observation data into a Tensor for the Reward Predictor.
    This handles both object and dictionary formats to prevent crashes.
    """
    # 1. Identify the record source
    record = obs_data.record_data if hasattr(obs_data, 'record_data') else obs_data.get('record_data', {})
    
    # 2. Extract specific fields safely
    diagnoses = record.diagnoses if hasattr(record, 'diagnoses') else record.get('diagnoses', [])
    prescriptions = record.prescriptions if hasattr(record, 'prescriptions') else record.get('prescriptions', [])
    raw_notes = record.raw_notes if hasattr(record, 'raw_notes') else record.get('raw_notes', "")
    current_view = obs_data.current_view if hasattr(obs_data, 'current_view') else obs_data.get('current_view', "STEP_0")

    # 3. Create the numerical feature vector
    features = [
        float(len(diagnoses)),
        float(len(prescriptions)),
        len(raw_notes) / 100.0,
        1.0 if "asthma" in raw_notes.lower() else 0.0,
        float(current_view.split("_")[-1]) if "_" in current_view else 0.0
    ]
    
    return torch.tensor(features, dtype=torch.float32)