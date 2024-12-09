import os

from HopfieldNetworkPyTorch.ModernHopfieldNetwork import ModernHopfieldNetwork, InteractionFunction
import torch
import torchvision
from args import get_args, get_config

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("CUDA is available")
else:
    print("CUDA is not available") 


def train_mnist(args, device=None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print("CUDA is available")
        else:
            print("CUDA is not available")
    
    # Define log paths
    log_paths = ["logs"]  
    
    # Create logs directory if it doesn't exist
    for path in log_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    # --------------------------------------------------------------------------------
    # Load MNIST dataset
    # --------------------------------------------------------------------------------
    
    datasetRoot: str = args.dataset_root

    mnistTrain = torchvision.datasets.MNIST(root=datasetRoot, download=True, train=True)
    mnistTest = torchvision.datasets.MNIST(root=datasetRoot, download=True, train=False)

    train = mnistTrain.data.reshape(-1, 28*28)
    train = 2*train/255.0 - 1 
    y_train = torch.nn.functional.one_hot(mnistTrain.targets)
    y_train = 2*y_train-1
    train = torch.concat((train, y_train), dim=1)
    train = train.type(torch.float64).T

    test = mnistTest.data.reshape(-1, 28*28)
    test = 2*test/255.0 - 1
    y_test = torch.nn.functional.one_hot(mnistTest.targets)
    y_test = 2*y_test-1
    test = torch.concat((test, y_test), dim=1)
    test = test.type(torch.float64).T

    # --------------------------------------------------------------------------------
    # Create Hopfield network of specific size
    # --------------------------------------------------------------------------------

    interactionVertex = args.interaction_vertex

    neuronMaskStartIndex = args.neuron_mask_start_index
    neuronMaskEndIndex = args.neuron_mask_end_index
    neuronMask = torch.arange(neuronMaskStartIndex, neuronMaskEndIndex)

    interactionFunc = InteractionFunction.LeakyRectifiedPolynomialInteractionFunction(n=interactionVertex)
    network = ModernHopfieldNetwork(args, device, interactionFunc)

    X = train[:, :10000].to(device) if args.debug else train.to(device)
    X_test = test[:, :10000].to(device) if args.debug else test.to(device)
    network.learnMemories(X,
                            X_test=X_test,
                            eval_interval=args.eval_interval,
                            log_paths=log_paths,
                            clampMemories=True,
                            neuronMask=neuronMask,
                            verbose=1,
                        )
    del X
    
if __name__ == "__main__":
    
    # --------------------------------------------------------------------------------
    # Manual configuration or load from config file
    # eg. args = get_config("configs/mnist_classification.yaml")
    # eg. args.threshold = 0.8
    # --------------------------------------------------------------------------------

    args = get_args()
    args = get_config("configs/mnist_classification.yaml")
    train_mnist(args, device)