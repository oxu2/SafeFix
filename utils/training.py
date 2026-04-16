"""
Training and evaluation utilities for machine learning models.
"""
import os
import re
import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, data_loader, device, args, criterion, save_preds=False):
    """
    Evaluate model performance on given data loader.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader to evaluate on
        device: Device to run evaluation on
        args: Arguments object containing dataset info
        criterion: Loss function
        save_preds: Whether to save predictions
        
    Returns:
        accuracy, loss (and predictions/labels if save_preds=True)
    """
    model.eval()
    total_correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            elif args.dataset == 'celeba':
                # For CelebA, ensure labels are for binary classification
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                total_correct += (preds == labels.int()).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

    acc = total_correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    
    if save_preds:
        return acc, np.array(all_preds), np.array(all_labels)
    return acc, avg_loss


def save_predictions(model, train_loader, valid_loader, device, args, criterion, 
                    train_idx, val_idx, modified_image_paths, modified_image_labels,
                    unlabeled_predictions, combined_labels, attributes, time_now):
    """
    Save model predictions to file.
    """
    print("Generating and saving predictions...")

    # Get predictions for training set
    _, train_preds, _ = evaluate(model, train_loader, device, args, criterion, save_preds=True)

    # Get predictions for validation set
    _, valid_preds, _ = evaluate(model, valid_loader, device, args, criterion, save_preds=True)

    if args.dataset == 'imagenet':
        original_count = len(train_idx)
        # Number of subsampled original predictions available
        sub_count = len(getattr(args, "selected_indices", []))
        # Subsampled original preds come first in train_preds
        sub_preds = train_preds[:sub_count]
        if getattr(args, 'pad_missing', False):
            # Initialize full-length original predictions with -1 (int) instead of NaN
            full_orig_preds = np.full((original_count,), -1, dtype=int)
            # Fill in the subsampled predictions at correct indices
            for idx, pred in zip(args.selected_indices, sub_preds):
                full_orig_preds[idx] = pred
            combined_preds = np.concatenate((full_orig_preds.flatten(), valid_preds.flatten()))
        else:
            combined_preds = np.concatenate((sub_preds.flatten(), valid_preds.flatten()))
    else:
        # Handle celeba or other cases
        if len(modified_image_paths) > 0:
            gen_cnt = len(modified_image_labels)
            if getattr(args, "pad_missing", False):
                # Pad missing original training predictions with -1 (int) instead of NaN
                full_orig_preds = np.full((args.orig_train_count,), -1, dtype=int)
                sub_preds = train_preds[:len(args.selected_indices)]
                for idx, pred in zip(args.selected_indices, sub_preds):
                    full_orig_preds[idx] = pred
                modified_preds = train_preds[len(args.selected_indices):]
                combined_preds = np.concatenate((
                    full_orig_preds.flatten(),
                    valid_preds.flatten(),
                    modified_preds.flatten(),
                    unlabeled_predictions
                ))
            else:
                original_train_preds = train_preds[:-gen_cnt]
                modified_preds = train_preds[-gen_cnt:]
                combined_preds = np.concatenate((
                    original_train_preds.flatten(),
                    valid_preds.flatten(),
                    modified_preds.flatten(),
                    unlabeled_predictions
                ))
        else:
            original_train_preds = train_preds
            combined_preds = np.concatenate((
                original_train_preds.flatten(),
                valid_preds.flatten(),
                unlabeled_predictions
            ))

    try:
        print(f"Combined predictions length: {len(combined_preds)} (should match combined_labels length: {len(combined_labels)})")
    except NameError:
        print(f"Combined predictions length: {len(combined_preds)}")
        
    # Determine filename suffix based on gen_only flag and ablation flag
    suffix = ""
    if args.gen_only:
        suffix += "_g"
    if args.no_controlnet:
        suffix += "_h" # Add 'h' for ablation/no_controlnet

    # Use simple naming if in simple ablation mode
    pred_key = "simple" if args.no_controlnet and args.simple else attributes
    np.save(
        f"save/pred/predictions_{args.dataset}_{args.model}_{pred_key}_{args.num_added_img}imgs{suffix}_{time_now}.npy",
        combined_preds
    )


def train_model(model, train_loader, valid_loader, criterion, optimizer, device, args, 
                scheduler=None, num_epochs=20, verbose=False):
    """
    Train a model with given parameters.
    
    Returns:
        training history (train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    """
    # Track best validation performance
    best_val_acc = 0.0
    best_model_state = None
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # For pretrained ViT and CLIP on ImageNet, freeze backbone for initial epochs
    freeze_epochs = 10
    if args.dataset == 'imagenet' and args.pretrained and args.model in ('vit', 'clip'):
        for name, param in model.named_parameters():
            # keep only classification head trainable
            if args.model == 'vit':
                if "head" not in name:
                    param.requires_grad = False
            else:  # clip
                if "classifier" not in name:
                    param.requires_grad = False
        if verbose:
            print(f"Froze {args.model} backbone parameters for initial {freeze_epochs} epochs")

    for epoch in range(num_epochs):
        # Unfreeze backbone after initial freeze_epochs for pretrained ImageNet CLIP/ViT
        if epoch == freeze_epochs and args.dataset == 'imagenet' and args.pretrained and args.model in ('vit', 'clip'):
            for param in model.parameters():
                param.requires_grad = True
            if verbose:
                print(f"Unfroze all {args.model} backbone parameters; now fine-tuning entire model")
        model.train()
        total_loss, total_correct = 0, 0
        
        for images, labels in train_loader:
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping only for CLIP and ViT models
                if args.model in ('clip', 'vit'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
            else:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping only for CLIP and ViT models
                if args.model in ('clip', 'vit'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                total_correct += ((torch.sigmoid(outputs) > 0.5).int() == labels.int()).sum().item()

        train_acc = total_correct / len(train_loader.dataset)
        val_acc, val_loss = evaluate(model, valid_loader, device, args, criterion)
        
        # Update best model if this epoch's validation accuracy is higher
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Step the learning rate scheduler if provided
        if scheduler:
            scheduler.step()
            if verbose:
                print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        # Compute average training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    # Generate plots if requested
    if args.plot:
        epochs = range(1, num_epochs+1)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0,0].plot(epochs, train_loss_history, marker='o')
        axes[0,0].set_title("Train Loss")
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("Loss")

        axes[0,1].plot(epochs, train_acc_history, marker='o')
        axes[0,1].set_title("Train Accuracy")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Accuracy")

        axes[1,0].plot(epochs, val_loss_history, marker='o')
        axes[1,0].set_title("Validation Loss")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Loss")

        axes[1,1].plot(epochs, val_acc_history, marker='o')
        axes[1,1].set_title("Validation Accuracy")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Accuracy")

        plt.tight_layout()
        attributes = args.attributes
        time_now = time.strftime("%m%d%H%M%S")
        plt.savefig(f"save/plot/plot_{args.model}_{attributes}_{args.num_added_img}imgs_{time_now}.png")
        if args.dataset == 'imagenet':
            plt.savefig(f"./plot_{args.dataset}_{args.model}_{attributes}_{args.num_added_img}imgs_{time_now}.png")
        plt.close()

    # Load best or last model before returning
    if args.save_pred_model == 'best' and best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Return training history for external use
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def parse_log(log_path):
    """
    Parse the log file to extract (attribute, description, accuracy) tuples.
    """
    # 1：ACC: 0.893557  ... attribute "beard", description "No"
    pattern1 = re.compile(
        r'ACC:\s*(?P<acc>\d+\.\d+).*attribute\s+"(?P<attr>[^"]+)",\s*description\s+"(?P<desc>[^"]+)"'
    )
    # 2：The accuracy with "hair color" = "white" is 0.9303...
    pattern2 = re.compile(
        r'The accuracy with\s+"(?P<attr>[^"]+)"\s*=\s*"(?P<desc>[^"]+)"\s*is\s*(?P<acc>\d+\.\d+)'
    )
    pattern_overall = re.compile(
        r'(?:Model Test ACC|Held-out test accuracy|Model Validation ACC):\s*(?P<acc>\d+\.\d+)'
    )
    overall_acc = None

    results = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if overall_acc is None:
                    m0 = pattern_overall.search(line)
                    if m0:
                        overall_acc = float(m0.group('acc'))
                m1 = pattern1.search(line)
                if m1:
                    groups = m1.groupdict()
                    results.append({
                        'attribute': groups['attr'],
                        'description': groups['desc'],
                        'accuracy': float(groups['acc'])
                    })
                    continue

                m2 = pattern2.search(line)
                if m2:
                    groups = m2.groupdict()
                    results.append({
                        'attribute': groups['attr'],
                        'description': groups['desc'],
                        'accuracy': float(groups['acc'])
                    })
    except FileNotFoundError:
        print(f"Warning: Comparison log file not found at {log_path}")
        return [], None

    return results, overall_acc
from utils.imports import *

# Simplified wrapper functions for backward compatibility
def evaluate(model, data_loader, save_preds=False, criterion=None, args=None, device=None):
    """Backward compatible evaluate function."""
    model.eval()
    total_correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    # Get globals if not provided (for backward compatibility)
    if args is None:
        import sys
        frame = sys._getframe(1)
        args = frame.f_globals.get('args')
        device = frame.f_globals.get('device') 
        criterion = frame.f_globals.get('criterion')

    with torch.no_grad():
        for images, labels in data_loader:
            if args.dataset == 'imagenet':
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            elif args.dataset == 'celeba':
                # For CelebA, ensure labels are for binary classification
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                total_correct += (preds == labels.int()).sum().item()
                if save_preds:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

    acc = total_correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader) if criterion else 0
    if save_preds:
        return acc, np.array(all_preds), np.array(all_labels)
    return acc, avg_loss


def save_predictions(model, train_loader, valid_loader, args=None, time_now=None, attributes=None, 
                    train_idx=None, val_idx=None, modified_image_labels=None, 
                    unlabeled_predictions=None):
    """Backward compatible save_predictions function."""
    print("Generating and saving predictions...")

    # Get globals if not provided (for backward compatibility)
    if args is None:
        import sys
        frame = sys._getframe(1)
        globals_dict = frame.f_globals
        args = globals_dict.get('args')
        time_now = time_now or globals_dict.get('time_now')
        attributes = attributes or globals_dict.get('attributes')
        train_idx = train_idx or globals_dict.get('train_idx')
        modified_image_labels = modified_image_labels or globals_dict.get('modified_image_labels', [])
        unlabeled_predictions = unlabeled_predictions or globals_dict.get('unlabeled_predictions')

    # Get predictions for training set
    _, train_preds, _ = evaluate(model, train_loader, save_preds=True)

    # Get predictions for validation set
    _, valid_preds, _ = evaluate(model, valid_loader, save_preds=True)

    if args.dataset == 'imagenet':
        original_count = len(train_idx)
        # Number of subsampled original predictions available
        sub_count = len(getattr(args, "selected_indices", []))
        # Subsampled original preds come first in train_preds
        sub_preds = train_preds[:sub_count]
        if getattr(args, 'pad_missing', False):
            # Initialize full-length original predictions with -1 (int) instead of NaN
            full_orig_preds = np.full((original_count,), -1, dtype=int)
            # Fill in the subsampled predictions at correct indices
            for idx, pred in zip(args.selected_indices, sub_preds):
                full_orig_preds[idx] = pred
            combined_preds = np.concatenate((full_orig_preds.flatten(), valid_preds.flatten()))
        else:
            combined_preds = np.concatenate((sub_preds.flatten(), valid_preds.flatten()))
    else:
        # Handle celeba or other cases
        if len(modified_image_labels) > 0:
            gen_cnt = len(modified_image_labels)
            if getattr(args, "pad_missing", False):
                # Pad missing original training predictions with -1 (int) instead of NaN
                full_orig_preds = np.full((args.orig_train_count,), -1, dtype=int)
                sub_preds = train_preds[:len(args.selected_indices)]
                for idx, pred in zip(args.selected_indices, sub_preds):
                    full_orig_preds[idx] = pred
                modified_preds = train_preds[len(args.selected_indices):]
                combined_preds = np.concatenate((
                    full_orig_preds.flatten(),
                    valid_preds.flatten(),
                    modified_preds.flatten(),
                    unlabeled_predictions
                ))
            else:
                original_train_preds = train_preds[:-gen_cnt]
                modified_preds = train_preds[-gen_cnt:]
                combined_preds = np.concatenate((
                    original_train_preds.flatten(),
                    valid_preds.flatten(),
                    modified_preds.flatten(),
                    unlabeled_predictions
                ))
        else:
            original_train_preds = train_preds
            combined_preds = np.concatenate((
                original_train_preds.flatten(),
                valid_preds.flatten(),
                unlabeled_predictions
            ))

    try:
        # This might not be available in all contexts
        print(f"Combined predictions length: {len(combined_preds)}")
    except:
        print(f"Combined predictions length: {len(combined_preds)}")
    
    # Determine filename suffix based on gen_only flag and ablation flag
    suffix = ""
    if args.gen_only:
        suffix += "_g"
    if args.no_controlnet:
        suffix += "_h" # Add 'h' for ablation/no_controlnet

    # Use simple naming if in simple ablation mode
    pred_key = "simple" if args.no_controlnet and args.simple else attributes
    np.save(
        f"save/pred/predictions_{args.dataset}_{args.model}_{pred_key}_{args.num_added_img}imgs{suffix}_{time_now}.npy",
        combined_preds
    )

    # Get predictions for validation set
    _, valid_preds, _ = evaluate(model, valid_loader, None, args, save_preds=True)

    if args.dataset == 'imagenet':
        original_count = len(train_idx)
        # Number of subsampled original predictions available
        sub_count = len(getattr(args, "selected_indices", []))
        # Subsampled original preds come first in train_preds
        sub_preds = train_preds[:sub_count]
        if getattr(args, 'pad_missing', False):
            # Initialize full-length original predictions with -1 (int) instead of NaN
            full_orig_preds = np.full((original_count,), -1, dtype=int)
            # Fill in the subsampled predictions at correct indices
            for idx, pred in zip(args.selected_indices, sub_preds):
                full_orig_preds[idx] = pred
            combined_preds = np.concatenate((full_orig_preds.flatten(), valid_preds.flatten()))
        else:
            combined_preds = np.concatenate((sub_preds.flatten(), valid_preds.flatten()))
    else:
        # Handle celeba or other cases
        if len(modified_image_labels) > 0:
            gen_cnt = len(modified_image_labels)
            if getattr(args, "pad_missing", False):
                # Pad missing original training predictions with -1 (int) instead of NaN
                full_orig_preds = np.full((args.orig_train_count,), -1, dtype=int)
                sub_preds = train_preds[:len(args.selected_indices)]
                for idx, pred in zip(args.selected_indices, sub_preds):
                    full_orig_preds[idx] = pred
                modified_preds = train_preds[len(args.selected_indices):]
                combined_preds = np.concatenate((
                    full_orig_preds.flatten(),
                    valid_preds.flatten(),
                    modified_preds.flatten(),
                    unlabeled_predictions
                ))
            else:
                original_train_preds = train_preds[:-gen_cnt]
                modified_preds = train_preds[-gen_cnt:]
                combined_preds = np.concatenate((
                    original_train_preds.flatten(),
                    valid_preds.flatten(),
                    modified_preds.flatten(),
                    unlabeled_predictions
                ))
        else:
            original_train_preds = train_preds
            combined_preds = np.concatenate((
                original_train_preds.flatten(),
                valid_preds.flatten(),
                unlabeled_predictions
            ))

    try:
        # This might not be available in all contexts
        print(f"Combined predictions length: {len(combined_preds)}")
    except:
        print(f"Combined predictions length: {len(combined_preds)}")
    
    # Determine filename suffix based on gen_only flag and ablation flag
    suffix = ""
    if args.gen_only:
        suffix += "_g"
    if args.no_controlnet:
        suffix += "_h" # Add 'h' for ablation/no_controlnet

    # Use simple naming if in simple ablation mode
    pred_key = "simple" if args.no_controlnet and args.simple else attributes
    np.save(
        f"save/pred/predictions_{args.dataset}_{args.model}_{pred_key}_{args.num_added_img}imgs{suffix}_{time_now}.npy",
        combined_preds
    )


def train_model(model, train_loader, valid_loader, criterion, optimizer, args, scheduler=None, num_epochs=20):
    # Track best validation performance
    best_val_acc = 0.0
    best_model_state = None
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # For pretrained ViT and CLIP on ImageNet, freeze backbone for initial epochs
    freeze_epochs = 10
    if args.dataset == 'imagenet' and args.pretrained and args.model in ('vit', 'clip'):
        for name, param in model.named_parameters():
            # keep only classification head trainable
            if args.model == 'vit':
                if "head" not in name:
                    param.requires_grad = False
            else:  # clip
                if "classifier" not in name:
                    param.requires_grad = False
        if args.verbose:
            print(f"Froze {args.model} backbone parameters for initial {freeze_epochs} epochs")

    for epoch in range(num_epochs):
        # Unfreeze backbone after initial freeze_epochs for pretrained ImageNet CLIP/ViT
        if epoch == freeze_epochs and args.dataset == 'imagenet' and args.pretrained and args.model in ('vit', 'clip'):
            for param in model.parameters():
                param.requires_grad = True
            if args.verbose:
                print(f"Unfroze all {args.model} backbone parameters; now fine-tuning entire model")
        model.train()
        total_loss, total_correct = 0, 0
        
        for images, labels in train_loader:
            if args.dataset == 'imagenet':
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping only for CLIP and ViT models
                if args.model in ('clip', 'vit'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
            else:
                images, labels = images.to(args.device), labels.to(args.device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Apply gradient clipping only for CLIP and ViT models
                if args.model in ('clip', 'vit'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                total_correct += ((torch.sigmoid(outputs) > 0.5).int() == labels.int()).sum().item()

        train_acc = total_correct / len(train_loader.dataset)
        val_acc, val_loss = evaluate(model, valid_loader, criterion, args)
        # Update best model if this epoch's validation accuracy is higher
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Step the learning rate scheduler if provided
        if scheduler:
            scheduler.step()
            if args.verbose:
                print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        # Compute average training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    if args.plot:
        epochs = range(1, num_epochs+1)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0,0].plot(epochs, train_loss_history, marker='o')
        axes[0,0].set_title("Train Loss")
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("Loss")

        axes[0,1].plot(epochs, train_acc_history, marker='o')
        axes[0,1].set_title("Train Accuracy")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Accuracy")

        axes[1,0].plot(epochs, val_loss_history, marker='o')
        axes[1,0].set_title("Validation Loss")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Loss")

        axes[1,1].plot(epochs, val_acc_history, marker='o')
        axes[1,1].set_title("Validation Accuracy")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Accuracy")

        plt.tight_layout()
        plt.savefig(f"save/plot/plot_{args.model}_{args.attributes}_{args.num_added_img}imgs_{args.time_now}.png")
        if args.dataset == 'imagenet':
            plt.savefig(f"./plot_{args.dataset}_{args.model}_{args.attributes}_{args.num_added_img}imgs_{args.time_now}.png")
        plt.close()

    # Load best or last model before saving predictions
    if args.save_pred_model == 'best' and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Return training history for external use
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def parse_log(log_path):
    """
    Parse the log file to extract (attribute, description, accuracy) tuples.
    """
    # 1：ACC: 0.893557  ... attribute "beard", description "No"
    pattern1 = re.compile(
        r'ACC:\s*(?P<acc>\d+\.\d+).*attribute\s+"(?P<attr>[^"]+)",\s*description\s+"(?P<desc>[^"]+)"'
    )
    # 2：The accuracy with "hair color" = "white" is 0.9303...
    pattern2 = re.compile(
        r'The accuracy with\s+"(?P<attr>[^"]+)"\s*=\s*"(?P<desc>[^"]+)"\s*is\s*(?P<acc>\d+\.\d+)'
    )
    pattern_overall = re.compile(
        r'(?:Model Test ACC|Held-out test accuracy|Model Validation ACC):\s*(?P<acc>\d+\.\d+)'
    )
    overall_acc = None

    results = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if overall_acc is None:
                    m0 = pattern_overall.search(line)
                    if m0:
                        overall_acc = float(m0.group('acc'))
                m1 = pattern1.search(line)
                if m1:
                    groups = m1.groupdict()
                    results.append({
                        'attribute': groups['attr'],
                        'description': groups['desc'],
                        'accuracy': float(groups['acc'])
                    })
                    continue

                m2 = pattern2.search(line)
                if m2:
                    groups = m2.groupdict()
                    results.append({
                        'attribute': groups['attr'],
                        'description': groups['desc'],
                        'accuracy': float(groups['acc'])
                    })
    except FileNotFoundError:
        print(f"Warning: Comparison log file not found at {log_path}")
        return [], None

    return results, overall_acc
