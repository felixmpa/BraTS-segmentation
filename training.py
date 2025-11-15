import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json

from data_preprocessing import create_data_loaders
from models import get_model, CombinedLoss, DiceLoss

class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self, config_type='basic'):
        if config_type == 'basic':
            self.learning_rate = 0.001
            self.batch_size = 8
            self.epochs = 10
            self.optimizer = 'adam'
            self.scheduler = 'cosine'
            self.weight_decay = 1e-4
            self.loss_function = 'combined'
            self.early_stopping_patience = 5
        elif config_type == 'optimized':
            self.learning_rate = 0.0005
            self.batch_size = 16
            self.epochs = 30
            self.optimizer = 'adamw'
            self.scheduler = 'cosine_warm'
            self.weight_decay = 1e-5
            self.loss_function = 'combined'
            self.early_stopping_patience = 8
            self.gradient_clipping = 1.0
            self.label_smoothing = 0.1
        else:
            raise ValueError(f"Unknown config type: {config_type}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        self.save_best_model = True
        self.log_interval = 10

class Trainer:
    def __init__(self, model, config: TrainingConfig, experiment_name: str = None):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        
        # Setup experiment tracking
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = f"runs/{self.experiment_name}"
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup loss function
        self._setup_loss_function()
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
    def _setup_optimizer(self):
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
    
    def _setup_loss_function(self):
        if self.config.loss_function == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.loss_function == 'dice':
            self.criterion = DiceLoss()
        elif self.config.loss_function == 'combined':
            self.criterion = CombinedLoss()
    
    def _setup_scheduler(self):
        if self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler == 'cosine_warm':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        elif self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Handle VAE separately
            if hasattr(self.model, 'reparameterize'):  # VAE
                output, mu, logvar = self.model(data)
                recon_loss = self.criterion(output, target)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.01 * kl_loss  # Beta-VAE with beta=0.01
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            
            # Gradient clipping if specified
            if hasattr(self.config, 'gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            
            self.optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            accuracy = (pred == target).float().mean().item()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard
            if batch_idx % self.config.log_interval == 0:
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss_Step', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy_Step', accuracy, step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], step)
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self, val_loader, epoch):
        self.model.eval()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = len(val_loader)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                # Handle VAE separately
                if hasattr(self.model, 'reparameterize'):  # VAE
                    output, mu, logvar = self.model(data)
                    loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                pred = output.argmax(dim=1)
                accuracy = (pred == target).float().mean().item()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                
                # Collect predictions for detailed metrics
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        return avg_loss, avg_accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader):
        print(f"Iniciando entrenamiento en {self.device}")
        print(f"Parámetros del modelo: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate_epoch(val_loader, epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy_Epoch', train_acc, epoch)
            self.writer.add_scalar('Val/Loss_Epoch', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy_Epoch', val_acc, epoch)
            self.writer.add_scalar('Val/Precision', val_precision, epoch)
            self.writer.add_scalar('Val/Recall', val_recall, epoch)
            self.writer.add_scalar('Val/F1_Score', val_f1, epoch)
            
            # Print epoch results
            print(f'Época {epoch+1}/{self.config.epochs}:')
            print(f'  Pérdida Entren.: {train_loss:.4f}, Precisión Entren.: {train_acc:.4f}')
            print(f'  Pérdida Val.: {val_loss:.4f}, Precisión Val.: {val_acc:.4f}')
            print(f'  Val Precisión: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
            
            # Early stopping and best model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                
                if self.config.save_best_model:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f'Parada temprana activada después de {epoch+1} épocas')
                    break
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            print('-' * 60)
        
        total_time = time.time() - start_time
        print(f'Entrenamiento completado en {total_time:.2f} segundos')
        print(f'Mejor pérdida de validación: {self.best_val_loss:.4f}')
        print(f'Mejor precisión de validación: {self.best_val_accuracy:.4f}')
        
        self.writer.close()
        return self.get_training_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, f'checkpoints/{self.experiment_name}_best.pth')
        else:
            torch.save(checkpoint, f'checkpoints/{self.experiment_name}_epoch_{epoch}.pth')
    
    def get_training_history(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }

class ExperimentRunner:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.results = {}
    
    def run_experiment(self, model_name: str, config_type: str = 'basic'):
        print(f"\n{'='*60}")
        print(f"Running {model_name} with {config_type} configuration")
        print(f"{'='*60}")
        
        # Create experiment name
        experiment_name = f"{model_name}_{config_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create configuration
        config = TrainingConfig(config_type)
        
        # Create data loaders
        train_loader, val_loader, data_analysis = create_data_loaders(
            self.csv_path, batch_size=config.batch_size
        )
        
        # Create model
        model = get_model(model_name)
        
        # Create trainer
        trainer = Trainer(model, config, experiment_name)
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        
        # Store results
        self.results[experiment_name] = {
            'model_name': model_name,
            'config_type': config_type,
            'history': history,
            'data_analysis': data_analysis,
            'experiment_name': experiment_name
        }
        
        return history
    
    def run_comparison_experiments(self):
        """Run comparison between basic and optimized configurations"""
        model_configs = [
            ('unet', 'basic'),
            ('unet', 'optimized'),
            ('resnet_unet', 'basic'),
            ('resnet_unet', 'optimized'),
            ('attention_unet', 'basic'),
            ('attention_unet', 'optimized'),
        ]
        
        for model_name, config_type in model_configs:
            try:
                self.run_experiment(model_name, config_type)
            except Exception as e:
                print(f"Error running {model_name} with {config_type}: {str(e)}")
                continue
        
        return self.results
    
    def save_results(self, filename: str = 'experiment_results.json'):
        """Save experiment results to file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for exp_name, result in self.results.items():
            serializable_result = result.copy()
            history = serializable_result['history']
            for key, value in history.items():
                if isinstance(value, (list, np.ndarray)):
                    history[key] = [float(v) if not isinstance(v, (list, dict)) else v for v in value]
            serializable_results[exp_name] = serializable_result
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Resultados guardados en {filename}")

def plot_training_curves(results_dict, save_path='training_curves.png'):
    """Plot training and validation curves for comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss curves
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[1, 0].set_title('Training Accuracy Comparison')
    axes[1, 1].set_title('Validation Accuracy Comparison')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (exp_name, result), color in zip(results_dict.items(), colors):
        history = result['history']
        epochs = range(1, len(history['train_losses']) + 1)
        
        label = f"{result['model_name']}_{result['config_type']}"
        
        axes[0, 0].plot(epochs, history['train_losses'], color=color, label=label, alpha=0.8)
        axes[0, 1].plot(epochs, history['val_losses'], color=color, label=label, alpha=0.8)
        axes[1, 0].plot(epochs, history['train_accuracies'], color=color, label=label, alpha=0.8)
        axes[1, 1].plot(epochs, history['val_accuracies'], color=color, label=label, alpha=0.8)
    
    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epoch')
    
    axes[0, 0].set_ylabel('Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_comparison(results_dict, save_path='performance_comparison.png'):
    """Plot final performance comparison"""
    models = []
    configs = []
    val_accuracies = []
    val_losses = []
    
    for exp_name, result in results_dict.items():
        models.append(result['model_name'])
        configs.append(result['config_type'])
        val_accuracies.append(result['history']['best_val_accuracy'])
        val_losses.append(result['history']['best_val_loss'])
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    x_pos = np.arange(len(models))
    bars1 = axes[0].bar(x_pos, val_accuracies, alpha=0.8)
    axes[0].set_xlabel('Experiments')
    axes[0].set_ylabel('Best Validation Accuracy')
    axes[0].set_title('Model Performance Comparison - Accuracy')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"{m}_{c}" for m, c in zip(models, configs)], rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, val_accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
    
    # Loss comparison
    bars2 = axes[1].bar(x_pos, val_losses, alpha=0.8, color='orange')
    axes[1].set_xlabel('Experiments')
    axes[1].set_ylabel('Best Validation Loss')
    axes[1].set_title('Model Performance Comparison - Loss')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"{m}_{c}" for m, c in zip(models, configs)], rotation=45)
    
    # Add value labels on bars
    for bar, loss in zip(bars2, val_losses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{loss:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    csv_path = "BraTS20 Training Metadata.csv"
    
    # Create experiment runner
    runner = ExperimentRunner(csv_path)
    
    # Run comparison experiments
    print("Iniciando comparación comprehensiva de modelos...")
    results = runner.run_comparison_experiments()
    
    # Save results
    runner.save_results()
    
    # Plot comparisons
    if results:
        plot_training_curves(results)
        plot_performance_comparison(results)
    
    print("¡Experimento completado exitosamente!")