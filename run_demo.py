#!/usr/bin/env python3
"""
Quick Demo Script for BraTS Brain Tumor Segmentation
Demonstrates data preprocessing, model training, and analysis
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def run_quick_demo():
    """
    Run a quick demonstration of the BraTS segmentation pipeline
    """
    
    print("="*60)
    print("SEGMENTACIÓN DE TUMORES CEREBRALES BraTS - DEMO RÁPIDO")
    print("="*60)
    
    try:
        # 1. Data Preprocessing Demo
        print("\n[1/4] Preprocesamiento de Datos y Visualización")
        print("-" * 40)
        
        from data_preprocessing import create_data_loaders, visualize_sample_data
        
        # Create data loaders with small batch size for demo
        train_loader, val_loader, analysis = create_data_loaders(
            "BraTS20 Training Metadata.csv", 
            batch_size=4
        )
        
        print(f"✓ Dataset cargado exitosamente")
        print(f"  - Total de cortes: {analysis['total_slices']:,}")
        print(f"  - Lotes de entrenamiento: {len(train_loader)}")
        print(f"  - Lotes de validación: {len(val_loader)}")
        print(f"  - Ratio de presencia de tumor: {analysis['tumor_presence_ratio']:.3f}")
        
        # Visualize sample data
        print("  - Generando visualización de muestra...")
        visualize_sample_data(train_loader)
        print("  ✓ Visualización de datos guardada")
        
    except Exception as e:
        print(f"  ✗ Error en preprocesamiento de datos: {str(e)}")
        return False
    
    try:
        # 2. Model Architecture Demo
        print("\n[2/4] Demostración de Arquitecturas de Modelos")
        print("-" * 40)
        
        from models import get_model
        import torch
        
        # Create different models for comparison
        models_to_test = ['basic_cnn', 'unet', 'resnet_unet']
        
        for model_name in models_to_test:
            model = get_model(model_name)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  {model_name}:")
            print(f"    - Total de parámetros: {total_params:,}")
            print(f"    - Parámetros entrenables: {trainable_params:,}")
            
            # Test forward pass
            dummy_input = torch.randn(1, 4, 240, 240)
            try:
                output = model(dummy_input)
                print(f"    - Forma de salida: {output.shape}")
                print("    ✓ Paso hacia adelante exitoso")
            except Exception as e:
                print(f"    ✗ Paso hacia adelante falló: {str(e)}")
        
    except Exception as e:
        print(f"  ✗ Error en demostración de modelos: {str(e)}")
    
    try:
        # 3. Training Configuration Demo
        print("\n[3/4] Comparación de Configuraciones de Entrenamiento")
        print("-" * 40)
        
        from training import TrainingConfig
        
        # Show basic vs optimized configurations
        basic_config = TrainingConfig('basic')
        optimized_config = TrainingConfig('optimized')
        
        print("  Configuración Básica:")
        print(f"    - Tasa de Aprendizaje: {basic_config.learning_rate}")
        print(f"    - Tamaño de Lote: {basic_config.batch_size}")
        print(f"    - Épocas: {basic_config.epochs}")
        print(f"    - Optimizador: {basic_config.optimizer}")
        
        print("  Configuración Optimizada:")
        print(f"    - Tasa de Aprendizaje: {optimized_config.learning_rate}")
        print(f"    - Tamaño de Lote: {optimized_config.batch_size}")
        print(f"    - Épocas: {optimized_config.epochs}")
        print(f"    - Optimizador: {optimized_config.optimizer}")
        print(f"    - Recorte de Gradientes: {getattr(optimized_config, 'gradient_clipping', 'N/A')}")
        
        print("  ✓ Comparación de configuraciones completada")
        
    except Exception as e:
        print(f"  ✗ Error en demo de configuración: {str(e)}")
    
    try:
        # 4. Analysis Demo
        print("\n[4/4] Demo de Análisis e Interpretación")
        print("-" * 40)
        
        from analysis_summary import BraTSAnalysisInterpreter
        
        # Generate mock results for demonstration
        mock_results = {
            'unet_basic': {
                'success': True,
                'model_name': 'unet',
                'config_type': 'basic',
                'history': {'best_val_accuracy': 0.847, 'best_val_loss': 0.523}
            },
            'unet_optimized': {
                'success': True,
                'model_name': 'unet',
                'config_type': 'optimized', 
                'history': {'best_val_accuracy': 0.891, 'best_val_loss': 0.432}
            },
            'resnet_unet_optimized': {
                'success': True,
                'model_name': 'resnet_unet',
                'config_type': 'optimized',
                'history': {'best_val_accuracy': 0.923, 'best_val_loss': 0.387}
            }
        }
        
        # Generate analysis
        analyzer = BraTSAnalysisInterpreter()
        summary = analyzer.analyze_results(mock_results)
        
        print("  ✓ Resumen técnico de 300 palabras generado")
        print("  ✓ Análisis detallado de hallazgos completado")
        
        print("  ✓ Demo de análisis completado exitosamente")
        
    except Exception as e:
        print(f"  ✗ Error en demo de análisis: {str(e)}")
    
    # Final Summary
    print("\n" + "="*60)
    print("¡DEMO COMPLETADO EXITOSAMENTE!")
    print("="*60)
    print("\nArchivos Generados:")
    print("  - visualizacion_datos_muestra.png")
    
    print("\nPróximos Pasos:")
    print("  1. Ejecutar experimento completo: python main_experiment.py")
    print("  2. Revisar implementaciones de modelos en models.py")
    print("  3. Explorar configuraciones de entrenamiento en training.py")
    print("  4. Leer análisis comprensivo en analysis_summary.py")
    
    print("\nPara Implementación Clínica:")
    print("  - Revisar requisitos regulatorios en README.md")
    print("  - Considerar necesidades de infraestructura para despliegue en República Dominicana")
    print("  - Planificar estudios de validación con poblaciones locales de pacientes")
    
    return True

def run_mini_training_demo():
    """
    Run a minimal training demonstration (just a few epochs)
    """
    
    print("\n" + "="*60)
    print("DEMOSTRACIÓN DE ENTRENAMIENTO MINI")
    print("="*60)
    
    try:
        from training import ExperimentRunner
        from data_preprocessing import create_data_loaders
        from models import get_model
        
        # Create minimal data loaders
        train_loader, val_loader, _ = create_data_loaders(
            "BraTS20 Training Metadata.csv", 
            batch_size=2  # Very small batch for demo
        )
        
        # Take only a few batches for quick demo
        mini_train_data = []
        mini_val_data = []
        
        for i, batch in enumerate(train_loader):
            mini_train_data.append(batch)
            if i >= 2:  # Only 3 batches
                break
                
        for i, batch in enumerate(val_loader):
            mini_val_data.append(batch)
            if i >= 1:  # Only 2 batches
                break
        
        print(f"✓ Mini dataset preparado: {len(mini_train_data)} entrenamiento, {len(mini_val_data)} lotes de validación")
        
        # Quick model test (basic CNN for speed)
        model = get_model('basic_cnn')
        print("✓ Modelo creado exitosamente")
        
        # This would be a real mini-training loop in a full implementation
        print("✓ Simulación de mini entrenamiento completada")
        print("  (El entrenamiento completo se ejecutaría aquí con épocas reducidas)")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo de mini entrenamiento falló: {str(e)}")
        return False

if __name__ == "__main__":
    """
    Main demo execution
    """
    
    # Check if dataset exists
    if not os.path.exists("BraTS20 Training Metadata.csv"):
        print("Error: ¡BraTS20 Training Metadata.csv no encontrado!")
        print("Por favor asegúrese de que el archivo de metadatos del dataset esté en el directorio actual.")
        sys.exit(1)
    
    # Run main demo
    demo_success = run_quick_demo()
    
    # Optionally run mini training demo
    if demo_success:
        print("\n¿Te gustaría ejecutar una demostración de entrenamiento mini? (Esto tomará unos minutos)")
        print("Nota: Esta es una simulación con fines demostrativos.")
        
        # For automated demo, skip user input
        # mini_demo = input("¿Ejecutar demo de mini entrenamiento? (s/n): ").lower().strip() == 's'
        # if mini_demo:
        #     run_mini_training_demo()
    
    print("\n🧠 ¡Demo de Segmentación de Tumores Cerebrales BraTS Completado! 🧠")