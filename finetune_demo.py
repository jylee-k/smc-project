import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml

# --- Import your model wrappers ---
# This script assumes 'realtime_solo.py' and its dependencies are in the same folder.
try:
    from realtime_solo import LocalPANN, LocalVGGish, LocalAST
except ImportError:
    print("Error: Could not import models from realtime_solo.py.")
    print("Please make sure this script is in the same directory.")
    exit()

# --- Configuration ---
NUM_ORIGINAL_CLASSES = 527
NUM_NEW_CLASSES = 1
TOTAL_CLASSES = NUM_ORIGINAL_CLASSES + NUM_NEW_CLASSES # 528

# This is a dummy config for LocalVGGish, as it requires one
DUMMY_VGGISH_CFG = {
    'label_csv': 'ast/egs/audioset/class_labels_indices.csv',
    'full_ckpt': './ast/pretrained_models/finetuned_vggish.pt'
}


# --- 1. Helper Function to Freeze a Model ---

def freeze_model(model):
    """
    Freezes all parameters in the model by setting requires_grad = False.
    """
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model {model.__class__.__name__} frozen. All layers non-trainable.")


# --- 2. Finetuning Function for Each Model ---

def finetune_pann_model():
    """
    Loads a pretrained LocalPANN, freezes it, and replaces its 
    classifier head to add a new class.
    """
    print("\n" + "="*30)
    print("Attempting PANN Finetune...")
    try:
        # Load the base model from your wrapper
        # We access .model to get the *actual* SoundEventDetection model inside
        model = LocalPANN(device='cpu').model
        freeze_model(model)

        # Find the final classifier layer (assuming it's named 'fc')
        # This is a common name for the final "fully-connected" layer
        if not hasattr(model, 'fc'):
            print("Error: PANN model has no 'fc' attribute. Cannot finetune.")
            return None

        in_features = model.fc.in_features
        
        # Replace the head with a new, untrained layer
        model.fc = nn.Linear(in_features, TOTAL_CLASSES)
        
        print(f"SUCCESS: Replaced PANN 'fc' layer.")
        print(f"New classifier head: nn.Linear({in_features}, {TOTAL_CLASSES}) (Trainable)")
        return model

    except Exception as e:
        print(f"FAILED to finetune PANN: {e}")
        return None

def finetune_vggish_model():
    """
    Loads a pretrained LocalVGGish, freezes it, and replaces its 
    classifier head to add a new class.
    """
    print("\n" + "="*30)
    print("Attempting VGGish Finetune...")
    try:
        # Load the base model from your wrapper
        model = LocalVGGish(cfg=DUMMY_VGGISH_CFG, device='cpu').model
        freeze_model(model)

        # The VGGish classifier is a 3-layer nn.Sequential.
        # We need to replace the *last* layer.
        if not (hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential)):
             print("Error: VGGish model has no 'classifier' attribute. Cannot finetune.")
             return None
        
        # Get in_features from the old final layer
        in_features = model.classifier[2].in_features
        
        # Replace the head
        model.classifier[2] = nn.Linear(in_features, TOTAL_CLASSES)
        
        print(f"SUCCESS: Replaced VGGish 'classifier[2]' layer.")
        print(f"New classifier head: nn.Linear({in_features}, {TOTAL_CLASSES}) (Trainable)")
        return model

    except Exception as e:
        print(f"FAILED to finetune VGGish: {e}")
        return None

def finetune_ast_model():
    """
    Loads a pretrained LocalAST, freezes it, and replaces its 
    classifier head to add a new class.
    """
    print("\n" + "="*30)
    print("Attempting AST Finetune...")
    try:
        # Load the base model from your wrapper
        model = LocalAST(device='cpu').model
        freeze_model(model)

        # Your AST model is wrapped in nn.DataParallel, 
        # so we access the real model via .module
        if not hasattr(model, 'module') or not hasattr(model.module, 'mlp_head'):
            print("Error: AST model has no 'module.mlp_head' attribute. Cannot finetune.")
            return None

        # Get in_features from the old head
        # Assuming mlp_head is a simple nn.Linear
        in_features = model.module.mlp_head.in_features
        
        # Replace the head
        model.module.mlp_head = nn.Linear(in_features, TOTAL_CLASSES)
        
        print(f"SUCCESS: Replaced AST 'module.mlp_head' layer.")
        print(f"New classifier head: nn.Linear({in_features}, {TOTAL_CLASSES}) (Trainable)")
        return model

    except Exception as e:
        print(f"FAILED to finetune AST: {e}")
        return None


# --- 3. "For Show" Demonstration ---

if __name__ == "__main__":
    print("--- Finetuning Demonstration Script ---")
    print(f"This script will demonstrate adding {NUM_NEW_CLASSES} new class(es).")
    print(f"Total classes will be {TOTAL_CLASSES}.")

    # --- Run the finetuning functions ---
    
    # We only need to show one for the demo
    new_model = finetune_vggish_model()
    # You could uncomment these to show all three
    # finetune_pann_model()
    # finetune_ast_model()

    if new_model:
        print("\n" + "="*30)
        print("--- 'For Show' Training Loop (Conceptual) ---")
        
        # This optimizer will *only* see the new, trainable parameters
        # because all other layers have requires_grad=False.
        optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Optimizer created. It will only train the new classifier head.")
        print(f"Loss function: CrossEntropyLoss")

        print("\n--- Training Loop Pseudocode ---")
        print("""
# This is what a real training loop would look like.
# It is commented out because we have no data.

# BATCH_SIZE = 16
# NUM_EPOCHS = 5
# my_new_dataloader = ... # (A dataloader with audio for the new class)

# for epoch in range(NUM_EPOCHS):
#     print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
#     for audio_batch, labels_batch in my_new_dataloader:
#
#         # 1. Zero gradients
#         optimizer.zero_grad()
#
#         # 2. Get model predictions
#         # (Note: VGGish/AST need feature extraction first)
#         # features = feature_extractor(audio_batch)
#         # outputs = new_model(features) 
#
#         # 3. Calculate loss
#         # loss = criterion(outputs, labels_batch)
#
#         # 4. Backpropagation
#         # loss.backward()
#
#         # 5. Update weights (of the new head only)
#         # optimizer.step()
#
#     print("Epoch complete.")

# --- After training, save the new model ---
# torch.save(new_model.state_dict(), "finetuned_vggish_528_classes.pt")
# print("New model checkpoint saved.")
        """)