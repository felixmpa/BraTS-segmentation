#!/usr/bin/env python3
"""
Análisis Exploratorio de Datos (EDA) - Dataset BraTS 2020
Segmentación de Tumores Cerebrales con Deep Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class BraTSEDA:
    """Análisis Exploratorio de Datos para BraTS Dataset"""
    
    def __init__(self, csv_path: str, output_dir: str = "eda_graficos"):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cargar datos
        print("📊 Cargando dataset BraTS...")
        self.df = pd.read_csv(csv_path)
        print(f"✅ Dataset cargado: {len(self.df):,} registros")
        
        # Preprocesar datos
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocesamiento inicial de los datos"""
        # Crear columnas derivadas
        self.df['has_tumor'] = self.df['background_ratio'] < 1.0
        self.df['tumor_ratio'] = 1 - self.df['background_ratio']
        self.df['total_tumor_pixels'] = self.df['label1_pxl_cnt'] + self.df['label2_pxl_cnt']
        
        # Categorías de tumor
        self.df['tumor_category'] = pd.cut(
            self.df['tumor_ratio'], 
            bins=[0, 0.01, 0.05, 0.15, 1.0], 
            labels=['Sin tumor', 'Tumor pequeño', 'Tumor mediano', 'Tumor grande'],
            include_lowest=True
        )
        
        print(f"✅ Preprocesamiento completado")
    
    def analisis_general(self):
        """Análisis general del dataset"""
        print("\n" + "="*60)
        print("📊 ANÁLISIS GENERAL DEL DATASET")
        print("="*60)
        
        # Estadísticas básicas
        print(f"📈 Estadísticas Generales:")
        print(f"  • Total de cortes: {len(self.df):,}")
        print(f"  • Volúmenes únicos: {self.df['volume'].nunique():,}")
        print(f"  • Rango de volúmenes: {self.df['volume'].min()} - {self.df['volume'].max()}")
        print(f"  • Cortes por volumen: {len(self.df) / self.df['volume'].nunique():.1f} promedio")
        
        # Distribución de tumores
        tumor_stats = self.df['has_tumor'].value_counts()
        print(f"\n🎯 Distribución de Tumores:")
        print(f"  • Con tumor: {tumor_stats[True]:,} ({tumor_stats[True]/len(self.df)*100:.1f}%)")
        print(f"  • Sin tumor: {tumor_stats[False]:,} ({tumor_stats[False]/len(self.df)*100:.1f}%)")
        
        # Estadísticas de píxeles
        print(f"\n🔍 Estadísticas de Píxeles por Etiqueta:")
        for label in ['label1_pxl_cnt', 'label2_pxl_cnt']:
            non_zero = self.df[self.df[label] > 0][label]
            if len(non_zero) > 0:
                print(f"  • {label}: {non_zero.mean():.0f} ± {non_zero.std():.0f} píxeles (cuando presente)")
        
        return self.df.describe()
    
    def grafico_distribucion_tumores(self):
        """Gráfico de distribución de tumores"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distribución de Tumores en Dataset BraTS 2020', fontsize=16, fontweight='bold')
        
        # 1. Pie chart - Con/Sin tumor
        tumor_counts = self.df['has_tumor'].value_counts()
        labels = ['Sin Tumor', 'Con Tumor']
        colors = ['lightcoral', 'lightblue']
        
        axes[0, 0].pie(tumor_counts.values, labels=labels, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[0, 0].set_title('Distribución General: Con/Sin Tumor')
        
        # 2. Histograma - Ratio de tumor
        axes[0, 1].hist(self.df[self.df['has_tumor']]['tumor_ratio'], 
                        bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Distribución del Ratio de Tumor (solo cortes con tumor)')
        axes[0, 1].set_xlabel('Ratio de Tumor')
        axes[0, 1].set_ylabel('Frecuencia')
        
        # 3. Barras - Categorías de tumor
        tumor_cat_counts = self.df['tumor_category'].value_counts()
        axes[1, 0].bar(range(len(tumor_cat_counts)), tumor_cat_counts.values, 
                       color=['red', 'orange', 'yellow', 'green'])
        axes[1, 0].set_title('Distribución por Categorías de Tumor')
        axes[1, 0].set_xlabel('Categoría')
        axes[1, 0].set_ylabel('Número de Cortes')
        axes[1, 0].set_xticks(range(len(tumor_cat_counts)))
        axes[1, 0].set_xticklabels(tumor_cat_counts.index, rotation=45)
        
        # Añadir valores en las barras
        for i, v in enumerate(tumor_cat_counts.values):
            axes[1, 0].text(i, v + 500, f'{v:,}', ha='center', va='bottom')
        
        # 4. Box plot - Ratio por volumen (muestra)
        sample_volumes = np.random.choice(self.df['volume'].unique(), size=20, replace=False)
        sample_data = self.df[self.df['volume'].isin(sample_volumes)]
        
        sns.boxplot(data=sample_data, x='volume', y='tumor_ratio', ax=axes[1, 1])
        axes[1, 1].set_title('Variabilidad del Ratio de Tumor por Volumen (Muestra)')
        axes[1, 1].set_xlabel('ID Volumen')
        axes[1, 1].set_ylabel('Ratio de Tumor')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribucion_tumores.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráfico guardado: distribucion_tumores.png")
    
    def grafico_analisis_volumenes(self):
        """Análisis por volúmenes"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis por Volúmenes - Dataset BraTS 2020', fontsize=16, fontweight='bold')
        
        # Estadísticas por volumen
        vol_stats = self.df.groupby('volume').agg({
            'has_tumor': 'sum',
            'tumor_ratio': 'mean',
            'slice': 'count',
            'total_tumor_pixels': 'sum'
        }).reset_index()
        
        vol_stats.columns = ['volume', 'cortes_con_tumor', 'ratio_tumor_promedio', 
                           'total_cortes', 'total_pixeles_tumor']
        vol_stats['porcentaje_cortes_tumor'] = (vol_stats['cortes_con_tumor'] / vol_stats['total_cortes']) * 100
        
        # 1. Histograma - Cortes por volumen
        axes[0, 0].hist(vol_stats['total_cortes'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 0].set_title('Distribución de Cortes por Volumen')
        axes[0, 0].set_xlabel('Número de Cortes por Volumen')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].axvline(vol_stats['total_cortes'].mean(), color='red', linestyle='--', 
                          label=f'Media: {vol_stats["total_cortes"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Scatter - Cortes vs Cortes con tumor
        axes[0, 1].scatter(vol_stats['total_cortes'], vol_stats['cortes_con_tumor'], 
                          alpha=0.6, color='purple')
        axes[0, 1].set_title('Total Cortes vs Cortes con Tumor por Volumen')
        axes[0, 1].set_xlabel('Total de Cortes')
        axes[0, 1].set_ylabel('Cortes con Tumor')
        
        # Línea de tendencia
        z = np.polyfit(vol_stats['total_cortes'], vol_stats['cortes_con_tumor'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(vol_stats['total_cortes'], p(vol_stats['total_cortes']), 
                       "r--", alpha=0.8, label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.1f}')
        axes[0, 1].legend()
        
        # 3. Histograma - Porcentaje de cortes con tumor por volumen
        axes[1, 0].hist(vol_stats['porcentaje_cortes_tumor'], bins=30, alpha=0.7, 
                       color='orange', edgecolor='black')
        axes[1, 0].set_title('% de Cortes con Tumor por Volumen')
        axes[1, 0].set_xlabel('Porcentaje de Cortes con Tumor')
        axes[1, 0].set_ylabel('Número de Volúmenes')
        axes[1, 0].axvline(vol_stats['porcentaje_cortes_tumor'].mean(), color='red', 
                          linestyle='--', label=f'Media: {vol_stats["porcentaje_cortes_tumor"].mean():.1f}%')
        axes[1, 0].legend()
        
        # 4. Top 10 volúmenes con más tumor
        top_tumor_volumes = vol_stats.nlargest(10, 'total_pixeles_tumor')
        axes[1, 1].bar(range(len(top_tumor_volumes)), top_tumor_volumes['total_pixeles_tumor'], 
                      color='red', alpha=0.7)
        axes[1, 1].set_title('Top 10 Volúmenes con Más Píxeles de Tumor')
        axes[1, 1].set_xlabel('Ranking')
        axes[1, 1].set_ylabel('Total Píxeles de Tumor')
        axes[1, 1].set_xticks(range(len(top_tumor_volumes)))
        axes[1, 1].set_xticklabels([f'Vol {v}' for v in top_tumor_volumes['volume']], rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'analisis_volumenes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráfico guardado: analisis_volumenes.png")
        
        return vol_stats
    
    def grafico_analisis_cortes(self):
        """Análisis de distribución de cortes"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Distribución de Cortes - Dataset BraTS 2020', fontsize=16, fontweight='bold')
        
        # 1. Distribución de slices
        axes[0, 0].hist(self.df['slice'], bins=50, alpha=0.7, color='teal', edgecolor='black')
        axes[0, 0].set_title('Distribución de Número de Slice')
        axes[0, 0].set_xlabel('Número de Slice')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # 2. Relación slice vs presencia de tumor
        slice_tumor = self.df.groupby('slice')['has_tumor'].agg(['sum', 'count']).reset_index()
        slice_tumor['tumor_percentage'] = (slice_tumor['sum'] / slice_tumor['count']) * 100
        
        axes[0, 1].plot(slice_tumor['slice'], slice_tumor['tumor_percentage'], 
                       color='red', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Porcentaje de Presencia de Tumor por Slice')
        axes[0, 1].set_xlabel('Número de Slice')
        axes[0, 1].set_ylabel('% de Cortes con Tumor')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Heatmap - Slice vs Volume (muestra)
        sample_volumes = np.random.choice(self.df['volume'].unique(), size=20, replace=False)
        sample_data = self.df[self.df['volume'].isin(sample_volumes)]
        
        heatmap_data = sample_data.pivot_table(
            values='has_tumor', 
            index='slice', 
            columns='volume', 
            aggfunc='mean',
            fill_value=0
        )
        
        sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Presencia de Tumor'}, 
                   ax=axes[1, 0])
        axes[1, 0].set_title('Mapa de Calor: Presencia de Tumor por Slice y Volumen (Muestra)')
        axes[1, 0].set_xlabel('ID Volumen')
        axes[1, 0].set_ylabel('Número de Slice')
        
        # 4. Distribución de píxeles de tumor por slice
        tumor_slices = self.df[self.df['has_tumor']]
        axes[1, 1].scatter(tumor_slices['slice'], tumor_slices['total_tumor_pixels'], 
                          alpha=0.5, color='purple', s=10)
        axes[1, 1].set_title('Píxeles de Tumor por Slice')
        axes[1, 1].set_xlabel('Número de Slice')
        axes[1, 1].set_ylabel('Total Píxeles de Tumor')
        
        # Línea de tendencia
        z = np.polyfit(tumor_slices['slice'], tumor_slices['total_tumor_pixels'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(tumor_slices['slice'], p(tumor_slices['slice']), 
                       "r--", alpha=0.8, label=f'Tendencia')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'analisis_cortes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráfico guardado: analisis_cortes.png")
    
    def grafico_analisis_etiquetas(self):
        """Análisis de las diferentes etiquetas de tumor"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Etiquetas de Tumor - Dataset BraTS 2020', fontsize=16, fontweight='bold')
        
        # Solo cortes con tumor
        tumor_data = self.df[self.df['has_tumor']].copy()
        
        # 1. Distribución de píxeles por etiqueta
        labels_data = {
            'Etiqueta 1': tumor_data['label1_pxl_cnt'],
            'Etiqueta 2': tumor_data['label2_pxl_cnt']
        }
        
        axes[0, 0].hist([labels_data['Etiqueta 1'], labels_data['Etiqueta 2']], 
                       bins=50, alpha=0.7, label=['Etiqueta 1', 'Etiqueta 2'],
                       color=['blue', 'red'])
        axes[0, 0].set_title('Distribución de Píxeles por Etiqueta')
        axes[0, 0].set_xlabel('Número de Píxeles')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # 2. Correlación entre etiquetas
        axes[0, 1].scatter(tumor_data['label1_pxl_cnt'], tumor_data['label2_pxl_cnt'], 
                          alpha=0.5, color='green', s=10)
        axes[0, 1].set_title('Correlación entre Etiquetas 1 y 2')
        axes[0, 1].set_xlabel('Píxeles Etiqueta 1')
        axes[0, 1].set_ylabel('Píxeles Etiqueta 2')
        
        # Calcular correlación
        correlation = tumor_data['label1_pxl_cnt'].corr(tumor_data['label2_pxl_cnt'])
        axes[0, 1].text(0.05, 0.95, f'Correlación: {correlation:.3f}', 
                       transform=axes[0, 1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 3. Proporción de etiquetas
        tumor_data['label1_prop'] = tumor_data['label1_pxl_cnt'] / tumor_data['total_tumor_pixels']
        tumor_data['label2_prop'] = tumor_data['label2_pxl_cnt'] / tumor_data['total_tumor_pixels']
        
        axes[1, 0].hist([tumor_data['label1_prop'], tumor_data['label2_prop']], 
                       bins=30, alpha=0.7, label=['Prop. Etiqueta 1', 'Prop. Etiqueta 2'],
                       color=['blue', 'red'])
        axes[1, 0].set_title('Proporción de Cada Etiqueta en Tumor Total')
        axes[1, 0].set_xlabel('Proporción')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].legend()
        
        # 4. Box plot comparativo
        label_comparison = pd.DataFrame({
            'Píxeles': list(tumor_data['label1_pxl_cnt']) + list(tumor_data['label2_pxl_cnt']),
            'Etiqueta': ['Etiqueta 1'] * len(tumor_data) + ['Etiqueta 2'] * len(tumor_data)
        })
        
        sns.boxplot(data=label_comparison, x='Etiqueta', y='Píxeles', ax=axes[1, 1])
        axes[1, 1].set_title('Comparación de Distribución por Etiqueta')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'analisis_etiquetas.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráfico guardado: analisis_etiquetas.png")
    
    def grafico_estadisticas_resumen(self):
        """Gráfico de estadísticas resumidas"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Estadísticas Resumidas - Dataset BraTS 2020', fontsize=16, fontweight='bold')
        
        # 1. Métricas clave
        metrics = {
            'Total Cortes': f"{len(self.df):,}",
            'Volúmenes': f"{self.df['volume'].nunique():,}",
            'Con Tumor': f"{self.df['has_tumor'].sum():,}",
            'Sin Tumor': f"{(~self.df['has_tumor']).sum():,}",
            'Avg Cortes/Vol': f"{len(self.df)/self.df['volume'].nunique():.1f}"
        }
        
        axes[0, 0].axis('off')
        axes[0, 0].text(0.5, 0.5, '\n'.join([f'{k}: {v}' for k, v in metrics.items()]), 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        axes[0, 0].set_title('Métricas Clave')
        
        # 2. Distribución de background ratio
        axes[0, 1].hist(self.df['background_ratio'], bins=50, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Distribución Background Ratio')
        axes[0, 1].set_xlabel('Background Ratio')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].axvline(self.df['background_ratio'].mean(), color='red', linestyle='--', 
                          label=f'Media: {self.df["background_ratio"].mean():.3f}')
        axes[0, 1].legend()
        
        # 3. Distribución logarítmica de píxeles de tumor
        tumor_pixels = self.df[self.df['total_tumor_pixels'] > 0]['total_tumor_pixels']
        axes[0, 2].hist(np.log10(tumor_pixels + 1), bins=40, alpha=0.7, color='green')
        axes[0, 2].set_title('Distribución log10(Píxeles Tumor + 1)')
        axes[0, 2].set_xlabel('log10(Píxeles + 1)')
        axes[0, 2].set_ylabel('Frecuencia')
        
        # 4. Mapa de calor de correlaciones
        numeric_cols = ['slice', 'label1_pxl_cnt', 'label2_pxl_cnt', 'background_ratio', 
                       'tumor_ratio', 'total_tumor_pixels']
        correlation_matrix = self.df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Matriz de Correlación')
        
        # 5. Tendencia de tumor por posición de slice
        slice_stats = self.df.groupby('slice').agg({
            'has_tumor': 'mean',
            'tumor_ratio': 'mean'
        }).reset_index()
        
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(slice_stats['slice'], slice_stats['has_tumor'] * 100, 
                        'b-', label='% con Tumor', alpha=0.8)
        line2 = ax2.plot(slice_stats['slice'], slice_stats['tumor_ratio'] * 100, 
                        'r-', label='% Ratio Tumor', alpha=0.8)
        
        ax1.set_xlabel('Número de Slice')
        ax1.set_ylabel('% de Cortes con Tumor', color='b')
        ax2.set_ylabel('Ratio Promedio de Tumor (%)', color='r')
        ax1.set_title('Presencia de Tumor por Posición de Slice')
        
        # Combinar leyendas
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # 6. Distribución de volúmenes por cantidad de tumor
        vol_tumor_stats = self.df.groupby('volume')['has_tumor'].sum().reset_index()
        vol_tumor_stats.columns = ['volume', 'cortes_con_tumor']
        
        axes[1, 2].hist(vol_tumor_stats['cortes_con_tumor'], bins=30, alpha=0.7, color='purple')
        axes[1, 2].set_title('Distribución: Cortes con Tumor por Volumen')
        axes[1, 2].set_xlabel('Cortes con Tumor por Volumen')
        axes[1, 2].set_ylabel('Número de Volúmenes')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'estadisticas_resumen.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráfico guardado: estadisticas_resumen.png")
    
    def generar_reporte_eda(self):
        """Generar reporte completo del EDA"""
        
        print("\n" + "="*80)
        print("📊 GENERANDO REPORTE COMPLETO DE EDA")
        print("="*80)
        
        # 1. Análisis general
        general_stats = self.analisis_general()
        
        # 2. Generar todos los gráficos
        print("\n📈 Generando gráficos...")
        self.grafico_distribucion_tumores()
        vol_stats = self.grafico_analisis_volumenes()
        self.grafico_analisis_cortes()
        self.grafico_analisis_etiquetas()
        self.grafico_estadisticas_resumen()
        
        # 3. Guardar estadísticas en archivo
        with open(self.output_dir / 'estadisticas_generales.txt', 'w', encoding='utf-8') as f:
            f.write("ESTADÍSTICAS GENERALES - DATASET BraTS 2020\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total de registros: {len(self.df):,}\n")
            f.write(f"Volúmenes únicos: {self.df['volume'].nunique():,}\n")
            f.write(f"Cortes con tumor: {self.df['has_tumor'].sum():,} ({self.df['has_tumor'].mean()*100:.1f}%)\n")
            f.write(f"Cortes sin tumor: {(~self.df['has_tumor']).sum():,} ({(~self.df['has_tumor']).mean()*100:.1f}%)\n")
            f.write(f"\nRatio de tumor promedio: {self.df[self.df['has_tumor']]['tumor_ratio'].mean():.4f}\n")
            f.write(f"Píxeles de tumor promedio (cuando presente): {self.df[self.df['has_tumor']]['total_tumor_pixels'].mean():.0f}\n")
            f.write(f"\nCortes por volumen (promedio): {len(self.df)/self.df['volume'].nunique():.1f}\n")
            f.write(f"Rango de slices: {self.df['slice'].min()} - {self.df['slice'].max()}\n")
            
            f.write("\n\nESTADÍSTICAS DESCRIPTIVAS:\n")
            f.write("-" * 30 + "\n")
            f.write(str(general_stats))
        
        print(f"\n✅ Reporte EDA completado!")
        print(f"📁 Gráficos guardados en: {self.output_dir}/")
        print(f"📄 Estadísticas guardadas en: {self.output_dir}/estadisticas_generales.txt")
        
        # Resumen de archivos generados
        files_generated = list(self.output_dir.glob('*.png'))
        files_generated.append(self.output_dir / 'estadisticas_generales.txt')
        
        print(f"\n📋 Archivos generados ({len(files_generated)}):")
        for file in sorted(files_generated):
            print(f"  📊 {file.name}")

def main():
    """Función principal"""
    print("🧠 ANÁLISIS EXPLORATORIO DE DATOS - BraTS 2020")
    print("Segmentación de Tumores Cerebrales con Deep Learning")
    print("="*60)
    
    # Verificar que existe el archivo CSV
    csv_path = "BraTS20 Training Metadata.csv"
    if not Path(csv_path).exists():
        print(f"❌ Error: No se encontró el archivo {csv_path}")
        print("   Por favor asegúrese de que el archivo CSV esté en el directorio actual.")
        return
    
    try:
        # Crear instancia del EDA
        eda = BraTSEDA(csv_path)
        
        # Generar reporte completo
        eda.generar_reporte_eda()
        
        print(f"\n🎉 ¡EDA completado exitosamente!")
        print(f"🔍 Revise la carpeta 'eda_graficos' para ver todos los análisis visuales.")
        
    except Exception as e:
        print(f"❌ Error durante el EDA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()