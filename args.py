import argparse
import yaml
import os

def get_args():
    parser = argparse.ArgumentParser(description='Modern Hopfield Network MNIST Configuration')
    
    # Experiment configuration
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment. If not provided, results will not be saved. If provided, results will be saved in logs/saved_results/{experiment_name}')
    parser.add_argument('--experiment_description', type=str, default=None,
                        help='Description of the experiment')
    parser.add_argument('--datetime', type=bool, default=True,
                        help='Whether to append datetime string to experiment name')
    
    # Task configuration
    parser.add_argument('--task', type=str, default='classification',
                        help='Task type')
    parser.add_argument('--dataset_root', type=str, default='../datasets/MNIST',
                        help='Root directory for dataset')

    # Data processing
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Threshold for binarizing MNIST images')
    
    # Network configuration
    parser.add_argument('--n_memories', type=int, default=100,
                        help='Number of memories to store')
    parser.add_argument('--neuron_mask_start_index', type=int, default=784,
                        help='Start index for neuron mask')
    parser.add_argument('--neuron_mask_end_index', type=int, default=794,
                        help='End index for neuron mask')
    parser.add_argument('--initial_memories_mu', type=float, default=-0.3,
                        help='Mean for initial memories initialization')
    parser.add_argument('--initial_memories_sigma', type=float, default=0.3,
                        help='Standard deviation for initial memories initialization')
    parser.add_argument('--interaction_vertex', type=int, default=20,
                        help='Interaction vertex parameter')
    parser.add_argument('--initial_temperature', type=float, default=540.,
                        help='Initial temperature parameter')
    parser.add_argument('--final_temperature', type=float, default=540.,
                        help='Final temperature parameter')
    parser.add_argument('--item_batch_size', type=int, default=128,
                        help='Batch size for items')
    
    # Training configuration
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluation interval')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='Maximum number of epochs')
    parser.add_argument('--initial_lr', type=float, default=4e-2,
                        help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.998,
                        help='Learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.6,
                        help='Momentum value')
    parser.add_argument('--error_power', type=int, default=30,
                        help='Error power parameter')
    
    # Display configuration
    parser.add_argument('--num_display_memories', type=int, default=100,
                        help='Number of memories to display')
    
    # Debug configuration
    parser.add_argument('--debug', action='store_false',
                        help='Debug mode')
    
    # Precision configuration
    parser.add_argument('--precision', type=float, default=1.0e-30,
                        help='Precision value for gradient normalization')

    parser.add_argument('--temperature_ramp', type=int, default=200,
                        help='Number of epochs to ramp temperature from initial to final value')

    args = parser.parse_args()
    return args

def get_config(config_path):
    """
    Load configuration from a YAML file and return updated args
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        argparse.Namespace: Updated arguments
    """
    args = get_args()
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            args.__dict__.update(config_dict)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    print(f"Loaded configuration from {config_path}")
    return args 