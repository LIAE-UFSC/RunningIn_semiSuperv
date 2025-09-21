#!/usr/bin/env python3
"""
Compressor Prediction Plotting Functions

This module provides functions to plot compressor run-in predictions from classifier results.
It supports plotting both Y_score and Y_predict values with optional moving average filtering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional

# Import classifier names from pareto.py
# from utils.pareto import classifier_names

# Define Portuguese classifier names directly to avoid import issues
classifier_names = {"LogisticRegression": "Regressão logística", 
                    "DecisionTreeClassifier": "Árvore de decisão",
                    "KNeighborsClassifier": "K-vizinhos mais próximos",
                    "LinearSVM": "MVS linear",
                    "RBFSVM": "MVS FBR",
                    "RandomForest": "Floresta aleatória",
                    "NeuralNet": "Rede neural",
                    "AdaBoost": "AdaBoost",
                    "NaiveBayes": "Bayes ingênuo",
                    "QDA": "Análise discriminante quadrática"
                    }

# Define the order of classifiers as used in pareto.py
classifier_order = [
    "LogisticRegression",
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "LinearSVM",
    "RBFSVM",
    "RandomForest",
    "NeuralNet",
    "AdaBoost",
    "NaiveBayes",
    "QDA"
]


def order_classifiers(data_filtered: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Order classifiers according to the predefined order from pareto.py.
    
    Args:
        data_filtered: Dictionary of filtered data by classifier
    
    Returns:
        List of classifier names in the correct order
    """
    available_classifiers = set(data_filtered.keys())
    ordered_classifiers = [clf for clf in classifier_order if clf in available_classifiers]
    return ordered_classifiers


def create_figure_legend(fig, data_filtered, colors, test_units):
    """
    Create a figure-level legend with classifier names organized in 4 columns at the top.
    
    Args:
        fig: matplotlib Figure object
        data_filtered: Dictionary of filtered data by classifier
        colors: List of colors for classifiers
        test_units: List of test unit names
    """
    legend_elements = []
    
    # Get ordered list of classifiers
    ordered_classifiers = order_classifiers(data_filtered)
    
    # Create legend entries for each classifier (both train and test if applicable)
    for j, classifier in enumerate(ordered_classifiers):
        color = colors[j % len(colors)]
        display_name = classifier_names.get(classifier, classifier)
        
        # Add train legend entry
        train_line = Line2D([0], [0], color=color, linewidth=2, 
                           linestyle='-', label=f'{display_name}')
        legend_elements.append(train_line)
    
    # Create the legend with 4 columns at the top of the figure
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=6,
              columnspacing=0.8, handlelength=1.5)


def configure_matplotlib_a4():
    """
    Configure matplotlib for A4 portrait layout with Arial fonts.
    """
    # Set Arial as default font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    # Set font sizes for A4 layout
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['figure.titlesize'] = 9
    
    # Set figure DPI for better quality
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    """
    Configure matplotlib for A4 portrait layout with Arial fonts.
    """
    # Set Arial as default font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    # Set font sizes for A4 layout
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['figure.titlesize'] = 9
    
    # Set figure DPI for better quality
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300


def calculate_a4_figure_size(n_rows: int, n_cols: int) -> Tuple[float, float]:
    """
    Calculate optimal figure size for A4 portrait with margins.
    
    Args:
        n_rows: Number of subplot rows
        n_cols: Number of subplot columns
    
    Returns:
        Tuple of (width, height) in inches
    """
    # A4 dimensions in inches: 8.27 x 11.69
    # With 2-3cm margins: effective area is about 6.5 x 9.5 inches
    a4_width = 6.5  # inches (about 16.5 cm)
    a4_height = 9.5  # inches (about 24 cm)
    
    # Calculate aspect ratio needed for the subplot grid
    subplot_aspect = n_cols / n_rows
    page_aspect = a4_width / a4_height
    
    if subplot_aspect > page_aspect:
        # Width-limited: use full width, scale height
        fig_width = a4_width
        fig_height = a4_width / subplot_aspect
        # Ensure we don't exceed page height
        if fig_height > a4_height:
            fig_height = a4_height
            fig_width = a4_height * subplot_aspect
    else:
        # Height-limited: use full height, scale width
        fig_height = a4_height
        fig_width = a4_height * subplot_aspect
        # Ensure we don't exceed page width
        if fig_width > a4_width:
            fig_width = a4_width
            fig_height = a4_width / subplot_aspect
    
    return fig_width, fig_height


def load_compressor_data(data_path: str, model: str) -> Dict[str, pd.DataFrame]:
    """
    Load and combine test and train data for all classifiers for a given compressor model.
    
    Args:
        data_path: Path to the directory containing the CSV files
        model: Compressor model ('a', 'b', or 'all')
    
    Returns:
        Dictionary with classifier names as keys and combined DataFrames as values
    """
    data = {}
    data_types = ['test', 'train']
    
    for data_type in data_types:
        pattern = os.path.join(data_path, f'data_{data_type}_RunIn_*_{model}.csv')
        files = glob.glob(pattern)
        
        for file in files:
            # Extract classifier name from filename
            # Pattern: data_{test/train}_RunIn_{classifier}_{model}.csv
            filename = os.path.basename(file)
            classifier = filename.replace(f'data_{data_type}_RunIn_', '').replace(f'_{model}.csv', '')
            
            # Initialize classifier dict if not exists
            if classifier not in data:
                data[classifier] = {}
            
            try:
                df = pd.read_csv(file)
                # Add a column to identify data type
                df['data_type'] = data_type
                data[classifier][data_type] = df
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    # Combine test and train data for each classifier
    data_combined = {}
    for classifier in data.keys():
        # Check if both test and train data exist
        if 'test' in data[classifier] and 'train' in data[classifier]:
            # Combine test and train data
            combined_df = pd.concat([
                data[classifier]['train'],
                data[classifier]['test']
            ], ignore_index=True)
            data_combined[classifier] = combined_df
        else:
            print(f"Warning: Missing data for {classifier}, model {model}")
    
    return data_combined


def filter_first_test(data_combined: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Filter data to include only the first test (N_ensaio == 0).
    
    Args:
        data_combined: Dictionary with classifier DataFrames
    
    Returns:
        Filtered dictionary with only N_ensaio == 0 data
    """
    data_filtered = {}
    for classifier, df in data_combined.items():
        filtered_df = df[df['N_ensaio'] == 0].copy()
        data_filtered[classifier] = filtered_df
    
    return data_filtered


def apply_moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a non-centered moving average filter.
    The first filtered sample starts when the first window is completely filled.
    
    Args:
        values: Input array
        window_size: Size of the moving average window
    
    Returns:
        Filtered array with NaN values for the first (window_size-1) samples
    """
    if window_size <= 1:
        return values
    
    filtered = np.full_like(values, np.nan, dtype=float)
    
    for i in range(window_size - 1, len(values)):
        filtered[i] = np.mean(values[i - window_size + 1:i + 1])
    
    return filtered


def get_unit_configuration(model: str, all_units: List[str]) -> Tuple[List[str], List[str]]:
    """
    Determine train and test units based on the compressor model.
    
    Args:
        model: Compressor model ('a', 'b', or 'all')
        all_units: List of all available units
    
    Returns:
        Tuple of (train_units, test_units)
    """
    if model == 'a':
        test_units = ['a3']
    elif model == 'b':
        test_units = ['b10']
    elif model == 'all':
        test_units = ['a3', 'b10']
    else:
        # Default: assume last unit alphabetically is test
        test_units = [sorted(all_units)[-1]]
    
    train_units = [unit for unit in all_units if unit not in test_units]
    
    return train_units, test_units


def calculate_grid_layout(n_units: int, plots_per_row: Optional[int] = None) -> Tuple[int, int]:
    """
    Calculate optimal grid layout for subplots.
    
    Args:
        n_units: Number of units to plot
        plots_per_row: Optional fixed number of plots per row
    
    Returns:
        Tuple of (n_rows, n_cols)
    """
    if plots_per_row is not None:
        n_cols = plots_per_row
        n_rows = (n_units + n_cols - 1) // n_cols  # Ceiling division
    else:
        # Default layouts based on model
        if n_units <= 6:
            n_cols = 3
        elif n_units <= 12:
            n_cols = 4
        else:
            n_cols = 5
        n_rows = (n_units + n_cols - 1) // n_cols
    
    return n_rows, n_cols


def plot_compressor_scores(data_path: str, model: str, 
                          plots_per_row: Optional[int] = None,
                          moving_avg_window: Optional[int] = None) -> plt.Figure:
    """
    Plot Y_score values for compressor prediction data.
    
    Args:
        data_path: Path to the directory containing CSV files
        model: Compressor model ('a', 'b', or 'all')
        plots_per_row: Number of plots per row (optional)
        moving_avg_window: Window size for moving average filter (optional)
    
    Returns:
        matplotlib Figure object
    """
    # Load and filter data
    data_combined = load_compressor_data(data_path, model)
    data_filtered = filter_first_test(data_combined)
    
    if not data_filtered:
        raise ValueError(f"No data found for model {model}")
    
    # Get all unique units
    all_units = set()
    for classifier, df in data_filtered.items():
        all_units.update(df['Unidade'].unique())
    
    all_units = sorted(list(all_units))
    train_units, test_units = get_unit_configuration(model, all_units)
    
    # Calculate grid layout
    all_units_to_plot = train_units + test_units
    n_units = len(all_units_to_plot)
    n_rows, n_cols = calculate_grid_layout(n_units, plots_per_row)
    
    # Calculate minimum duration for x-axis
    min_duration = float('inf')
    for classifier, df in data_filtered.items():
        for unit in all_units:
            unit_data = df[df['Unidade'] == unit]
            if len(unit_data) > 0:
                unit_duration = unit_data['Tempo'].max()
                min_duration = min(min_duration, unit_duration)
    
    # Define colors for different classifiers
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Configure matplotlib for A4 layout
    configure_matplotlib_a4()
    
    # Calculate A4-optimized figure size
    fig_width, fig_height = calculate_a4_figure_size(n_rows, n_cols)
    
    # Create figure and subplots with A4-optimized size
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Plot data for each unit
    for i, unit in enumerate(all_units_to_plot):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        is_test_unit = unit in test_units
        true_label_plotted = False
        
        # Get ordered list of classifiers
        ordered_classifiers = order_classifiers(data_filtered)
        
        for j, classifier in enumerate(ordered_classifiers):
            df = data_filtered[classifier]
            unit_df = df[df['Unidade'] == unit].sort_values('Tempo')
            
            if len(unit_df) == 0:
                continue
            
            # Separate test and train data
            train_df = unit_df[unit_df['data_type'] == 'train'].sort_values('Tempo')
            test_df = unit_df[unit_df['data_type'] == 'test'].sort_values('Tempo')
            
            color = colors[j % len(colors)]
            
            # Plot train data (solid line)
            if len(train_df) > 0:
                y_values = train_df['Y_score'].values
                if moving_avg_window:
                    y_values = apply_moving_average(y_values, moving_avg_window)
                
                # Use Portuguese name if available, otherwise use original classifier name
                display_name = classifier_names.get(classifier, classifier)
                ax.plot(train_df['Tempo'], np.array(y_values)*100, 
                       label=f'{display_name}', 
                       color=color, linewidth=2, alpha=0.8, linestyle='-')
            
            # Plot test data (dashed line) - only for test units
            if len(test_df) > 0 and is_test_unit:
                y_values = test_df['Y_score'].values
                if moving_avg_window:
                    y_values = apply_moving_average(y_values, moving_avg_window)
                
                # Use Portuguese name if available, otherwise use original classifier name
                display_name = classifier_names.get(classifier, classifier)
                ax.plot(test_df['Tempo'], np.array(y_values)*100, 
                       label=f'{display_name} (Test)', 
                       color=color, linewidth=2, alpha=0.8, linestyle='-')
            
            # Plot true labels only once
            if not true_label_plotted and len(unit_df) > 0:
                tempo_all = unit_df['Tempo'].sort_values()
                y_true_all = unit_df.set_index('Tempo')['Y_true'].loc[tempo_all]
                
                # Shade background gray where Y_true == 0
                tempo_values = tempo_all.values
                y_true_values = y_true_all.values
                
                # Find continuous segments where Y_true == 0
                segments = []
                start_idx = None
                for idx in range(len(y_true_values)):
                    if y_true_values[idx] == 0:
                        if start_idx is None:
                            start_idx = idx
                    else:
                        if start_idx is not None:
                            segments.append((start_idx, idx - 1))
                            start_idx = None
                
                # Handle case where last segment extends to the end
                if start_idx is not None:
                    segments.append((start_idx, len(y_true_values) - 1))
                
                # Shade the segments
                for start_idx, end_idx in segments:
                    ax.axvspan(0, tempo_values[end_idx], 
                              color='gray', alpha=0.2)
                
                true_label_plotted = True
        
        # Set x-axis limits to shortest test duration
        ax.set_xlim(0, min_duration)
        
        # Add unit name as text box inside the plot
        unit_color = 'lightblue' if unit.startswith('a') else 'lightgreen'
        if unit in test_units:
            unit_color = 'lightyellow'
            
        ax.text(0.98, 0.02, unit.upper(), transform=ax.transAxes, 
               fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor=unit_color, alpha=0.8),
               verticalalignment='bottom', horizontalalignment='right')
        
        # Only show y-label and y-ticks on leftmost plots of each row
        col = i % n_cols
        if col == 0:  # Leftmost column
            ax.set_ylabel('Prob. de Amaciamento')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 100.5)
    
    # Create figure-level legend at the top
    create_figure_legend(fig, data_filtered, colors, test_units)
    
    # Set x-label for bottom row plots
    for i in range(len(all_units_to_plot)):
        row = i // n_cols
        if row == n_rows - 1:  # Bottom row
            ax_bottom = plt.subplot(n_rows, n_cols, i + 1)
            ax_bottom.set_xlabel('Tempo [h]')
    
    plt.tight_layout(rect=(0, 0, 1, 0.90))  # Leave space for legend at top
    
    return fig


def plot_compressor_predictions(data_path: str, model: str, 
                               plots_per_row: Optional[int] = None,
                               moving_avg_window: Optional[int] = None) -> plt.Figure:
    """
    Plot Y_predict values for compressor prediction data.
    
    Args:
        data_path: Path to the directory containing CSV files
        model: Compressor model ('a', 'b', or 'all')
        plots_per_row: Number of plots per row (optional)
        moving_avg_window: Window size for moving average filter (optional)
    
    Returns:
        matplotlib Figure object
    """
    # Load and filter data
    data_combined = load_compressor_data(data_path, model)
    data_filtered = filter_first_test(data_combined)
    
    if not data_filtered:
        raise ValueError(f"No data found for model {model}")
    
    # Get all unique units
    all_units = set()
    for classifier, df in data_filtered.items():
        all_units.update(df['Unidade'].unique())
    
    all_units = sorted(list(all_units))
    train_units, test_units = get_unit_configuration(model, all_units)
    
    # Calculate grid layout
    all_units_to_plot = train_units + test_units
    n_units = len(all_units_to_plot)
    n_rows, n_cols = calculate_grid_layout(n_units, plots_per_row)
    
    # Calculate minimum duration for x-axis
    min_duration = float('inf')
    for classifier, df in data_filtered.items():
        for unit in all_units:
            unit_data = df[df['Unidade'] == unit]
            if len(unit_data) > 0:
                unit_duration = unit_data['Tempo'].max()
                min_duration = min(min_duration, unit_duration)
    
    # Define colors for different classifiers
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Configure matplotlib for A4 layout and create figure
    configure_matplotlib_a4()
    fig_width, fig_height = calculate_a4_figure_size(n_rows, n_cols)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Plot data for each unit
    for i, unit in enumerate(all_units_to_plot):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        is_test_unit = unit in test_units
        true_label_plotted = False
        
        # Get ordered list of classifiers
        ordered_classifiers = order_classifiers(data_filtered)
        
        for j, classifier in enumerate(ordered_classifiers):
            df = data_filtered[classifier]
            unit_df = df[df['Unidade'] == unit].sort_values('Tempo')
            
            if len(unit_df) == 0:
                continue
            
            # Separate test and train data
            train_df = unit_df[unit_df['data_type'] == 'train'].sort_values('Tempo')
            test_df = unit_df[unit_df['data_type'] == 'test'].sort_values('Tempo')
            
            color = colors[j % len(colors)]
            
            # Plot train data (solid line)
            if len(train_df) > 0:
                y_values = train_df['Y_predict'].values.astype(float)
                if moving_avg_window:
                    y_values = apply_moving_average(y_values, moving_avg_window)
                
                # Use Portuguese name if available, otherwise use original classifier name
                display_name = classifier_names.get(classifier, classifier)
                ax.plot(train_df['Tempo'], np.array(y_values)*100, 
                       label=f'{display_name}', 
                       color=color, linewidth=2, alpha=0.8, linestyle='-')
            
            # Plot test data (dashed line) - only for test units
            if len(test_df) > 0 and is_test_unit:
                y_values = test_df['Y_predict'].values.astype(float)
                if moving_avg_window:
                    y_values = apply_moving_average(y_values, moving_avg_window)
                
                # Use Portuguese name if available, otherwise use original classifier name
                display_name = classifier_names.get(classifier, classifier)
                ax.plot(test_df['Tempo'], np.array(y_values)*100, 
                       label=f'{display_name} (Test)', 
                       color=color, linewidth=2, alpha=0.8, linestyle='-')
            
            # Plot true labels only once
            if not true_label_plotted and len(unit_df) > 0:
                tempo_all = unit_df['Tempo'].sort_values()
                y_true_all = unit_df.set_index('Tempo')['Y_true'].loc[tempo_all]
                
                # Shade background gray where Y_true == 0
                tempo_values = tempo_all.values
                y_true_values = y_true_all.values
                
                # Find continuous segments where Y_true == 0
                segments = []
                start_idx = None
                for idx in range(len(y_true_values)):
                    if y_true_values[idx] == 0:
                        if start_idx is None:
                            start_idx = idx
                    else:
                        if start_idx is not None:
                            segments.append((start_idx, idx - 1))
                            start_idx = None
                
                # Handle case where last segment extends to the end
                if start_idx is not None:
                    segments.append((start_idx, len(y_true_values) - 1))
                
                # Shade the segments
                for start_idx, end_idx in segments:
                    ax.axvspan(0, tempo_values[end_idx], 
                              color='gray', alpha=0.2)
                
                true_label_plotted = True
        
        # Set x-axis limits to shortest test duration
        ax.set_xlim(0, min_duration)
        
        # Add unit name as text box inside the plot
        unit_color = 'lightblue' if unit.startswith('a') else 'lightgreen'
        if unit in test_units:
            unit_color = 'lightyellow'
            
        ax.text(0.98, 0.02, unit.upper(), transform=ax.transAxes, 
               fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor=unit_color, alpha=0.8),
               verticalalignment='bottom', horizontalalignment='right')
        
        # Only show y-label and y-ticks on leftmost plots of each row
        col = i % n_cols
        if col == 0:  # Leftmost column
            ax.set_ylabel('Prediction / Label')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 100.5)
    
    # Create figure-level legend at the top
    create_figure_legend(fig, data_filtered, colors, test_units)
    
    # Set x-label for bottom row plots
    for i in range(len(all_units_to_plot)):
        row = i // n_cols
        if row == n_rows - 1:  # Bottom row
            ax_bottom = plt.subplot(n_rows, n_cols, i + 1)
            ax_bottom.set_xlabel('Tempo [h]')
    
    plt.tight_layout(rect=(0, 0, 1, 0.90))  # Leave space for legend at top
    
    return fig


def main():
    """
    Main function with argument parsing for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Plot compressor prediction results')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to directory containing CSV data files')
    parser.add_argument('--model', type=str, choices=['a', 'b', 'all'], required=True,
                       help='Compressor model to plot (a, b, or all)')
    parser.add_argument('--plot_type', type=str, choices=['scores', 'predictions', 'both'], 
                       default='both', help='Type of plot to generate')
    parser.add_argument('--plots_per_row', type=int, default=None,
                       help='Number of plots per row (default: auto)')
    parser.add_argument('--moving_avg_window', type=int, default=1,
                       help='Window size for moving average filter (default: 1, no filtering)')
    parser.add_argument('--output_dir', type=str, default=os.path.join('Results', 'time_figs'),
                       help='Output directory for PDF files (default: Results/time_figs)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.plot_type in ['scores', 'both']:
            print(f"Generating score plots for model {args.model}...")
            fig_scores = plot_compressor_scores(
                args.data_path, 
                args.model, 
                args.plots_per_row,
                args.moving_avg_window
            )
            
            # Save figure
            output_file = os.path.join(args.output_dir, f'compressor_{args.model}_scores_{args.moving_avg_window}.pdf')
            fig_scores.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Score plot saved to: {output_file}")
            plt.close(fig_scores)
        
        if args.plot_type in ['predictions', 'both']:
            print(f"Generating prediction plots for model {args.model}...")
            fig_predictions = plot_compressor_predictions(
                args.data_path, 
                args.model, 
                args.plots_per_row,
                args.moving_avg_window
            )
            
            # Save figure
            output_file = os.path.join(args.output_dir, f'compressor_{args.model}_predictions_{args.moving_avg_window}.pdf')
            fig_predictions.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Prediction plot saved to: {output_file}")
            plt.close(fig_predictions)
        
        print("All plots generated successfully!")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())