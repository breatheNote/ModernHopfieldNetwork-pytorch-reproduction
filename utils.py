import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os

def displayStatesAsImage(statesTensor: torch.Tensor, 
                        numImages: int, 
                        imageShape: tuple[int, int] = (28, 28),
                        fig_kw: dict = {},
                        title: str = None) -> tuple[Figure, list[Axes]]:
    """
    Given a set of tensors of shape (imageShape[0]*imageShape[1]*numClasses, N), 
    take only the image neurons of the first numImages items and display them.
    Values are clipped to [-1, 1] range where:
    - Red represents +1
    - White represents 0
    - Blue represents -1
    
    Args:
        statesTensor: Input tensor containing the states
        numImages: Number of images to display
        imageShape: Shape of each image (default: MNIST shape (28, 28))
        fig_kw: Additional keyword arguments for plt.subplots
        title: Title to display on the figure
    """
    plt.ion()  # Turn on interactive mode for display
    numSubplot = np.ceil(np.sqrt(numImages)).astype(int)
    fig, axes = plt.subplots(numSubplot, numSubplot, **fig_kw)
    if title:
        fig.suptitle(title, fontsize=16)  # 제목 크기 증가
        
    for ax in np.ravel(axes):
        ax.axis("off")
    
    # Create custom colormap: red -> white -> blue
    cmap = plt.cm.RdBu_r  # Red-White-Blue reversed colormap
    
    for itemIndex, ax in zip(range(numImages), np.ravel(axes)):
        targetMemory: np.ndarray = statesTensor[:imageShape[0]*imageShape[1], itemIndex].to("cpu").detach().numpy()
        targetMemory = np.clip(targetMemory, -1, 1)  
        targetMemory = targetMemory.reshape(imageShape)
        im = ax.imshow(targetMemory, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar to the right of the last figure
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Neuron State')
    
    return fig, axes 

def saveStatesAsImage(statesTensor: torch.Tensor,
                     numImages: int,
                     log_paths: list[str],
                     filename: str,
                     imageShape: tuple[int, int] = (28, 28),
                     fig_kw: dict = {},
                     dpi: int = 300,
                     title: str = None, 
                     autoScale: bool = False) -> None:
    """
    Save states as image to specified path.
    
    Args:
        statesTensor: Input tensor containing the states
        numImages: Number of images to display
        log_paths: List of paths to save the figure
        filename: Name of the file (without extension)
        imageShape: Shape of each image (default: MNIST shape (28, 28))
        fig_kw: Additional keyword arguments for plt.subplots
        dpi: DPI for the saved figure (default: 300)
        title: Title to display on the figure
        autoScale: Whether to automatically scale the color range to [-1, 1] (default: True)
    """
    plt.ioff()  # Turn off interactive mode for saving
    for log_path in log_paths:
        os.makedirs(log_path, exist_ok=True)
    
    numSubplot = np.ceil(np.sqrt(numImages)).astype(int)
    fig, axes = plt.subplots(numSubplot, numSubplot, **fig_kw)
    if title:
        fig.suptitle(title, fontsize=16)
    
    for ax in np.ravel(axes):
        ax.axis("off")
    
    # Create custom colormap: red -> white -> blue
    cmap = 'bwr' # before : plt.cm.RdBu_r
    
    for itemIndex, ax in zip(range(numImages), np.ravel(axes)):
        targetMemory: np.ndarray = statesTensor[:imageShape[0]*imageShape[1], itemIndex].to("cpu").detach().numpy()
        targetMemory = np.clip(targetMemory, -1, 1)
        if autoScale:
            targetMemory = targetMemory / np.max(np.abs(targetMemory))
        targetMemory = targetMemory.reshape(imageShape)
        im = ax.imshow(targetMemory, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar to the right of the last figure
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Neuron State')
    
    # Full path for saving
    for log_path in log_paths:
        full_path = os.path.join(log_path, f"{filename}_{'autoScale' if autoScale else 'fixedScale'}.png")
        
        # Save the figure
        fig.savefig(full_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)  # Close the figure to free memory
    

import os
import shutil
from pathlib import Path

def copy_script_to_logs(script_path, log_dir="logs"):
    """
    Copy the script to a log directory
    
    Args:
        script_path (str): Path to the script being executed
        log_dir (str): Directory to save the copy (default: 'logs')
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Get the script filename
    script_name = Path(script_path).name
    
    # Create the destination path
    log_file = os.path.join(log_dir, script_name)
    
    # Copy the file
    shutil.copy2(script_path, log_file)