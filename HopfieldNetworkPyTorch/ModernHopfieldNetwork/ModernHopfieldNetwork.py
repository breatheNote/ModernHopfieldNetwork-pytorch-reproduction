from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Callable

import torch

from HopfieldNetworkPyTorch import utils

from .InteractionFunction import AbstractInteractionFunction

import pandas as pd
import os

class ModernHopfieldNetwork():
    
    def __init__(self, args, torchDevice: str, interactionFunction: AbstractInteractionFunction):
        """
        Create a new modern Hopfield network with the specified configuration from args.
        Note the interaction function must implement InteractionFunction.AbstractInteractionFunction

        :param args: Arguments containing network configuration
        :param torchDevice: The pytorch device to store the memories on, e.g. "cpu" or "cuda"
        :param interactionFunction: An implementation of InteractionFunction.AbstractInteractionFunction
        """
        
        self.args = args
        self.dimension = args.dimension
        
        self.memories = torch.randn(size=(self.dimension, args.n_memories), device=torchDevice, dtype=torch.float64) * self.args.initial_memories_sigma + self.args.initial_memories_mu
        
        ''' before :
        self.memories = torch.randn(size=(self.dimension, args.n_memories), device=torchDevice, dtype=torch.float64).mul(0.1)
        self.memories.clamp(-1,1)
        '''
        self.memories.requires_grad_(True)
        self.interactionFunction = interactionFunction

        self.itemBatchSize = args.item_batch_size
        self.neuronBatchSize = args.neuron_batch_size if hasattr(args, 'neuron_batch_size') else None

    def setItemBatchSize(self, itemBatchSize: int) :
        self.itemBatchSize = itemBatchSize

    def setNeuronBatchSize(self, neuronBatchSize: int) :
        self.neuronBatchSize = neuronBatchSize

    def setMemories(self, memories: torch.Tensor):
        """
        Set the memories of the network directly. Note the memories must be moved to the preferred device before being passed.

        :param memories: The new memories of the network. Must be of shape (network.dimension, nMemories) and be moved to the preferred device.
        """
        if memories.shape != self.memories.shape:
            raise ValueError("memories should have shape (network.dimension, nMemories)")
        self.memories = memories.to(self.memories.device).requires_grad_()
    
    def learnMemories(self, X: torch.Tensor,
                     X_test: torch.Tensor = None,
                     eval_interval: int = None, 
                     clampMemories: bool = True, 
                     neuronMask: torch.Tensor = None,
                     verbose: int = 2,
                     log_paths: list = ["logs/current_results"],
                     num_display_memories: int = None, 
                     image_shape: tuple = None
                      ):
        """
        Stabilize a set of states X by gradient descent and back propagation.
        Configuration is taken from self.args.

        :param X: Training states, a tensor of shape (network.dimension, n)
        :param X_test: Test states, a tensor of shape (network.dimension, m)
        :param eval_interval: Override args.eval_interval if provided
        :param clampMemories: Boolean to clamp neuron values between -1 and 1
        :param neuronMask: Override which neurons to update during learning
        :param verbose: Verbosity level (0-2)
        :param log_paths: List of paths to save training progress
        :param num_display_memories: Override args.num_display_memories if provided
        :param image_shape: Override args.image_shape if provided
        """
        
        # Use args values or overrides if provided
        eval_interval = eval_interval if eval_interval is not None else self.args.eval_interval
        num_display_memories = num_display_memories if num_display_memories is not None else self.args.num_display_memories
        image_shape = image_shape if image_shape is not None else self.args.image_shape
        
        # Use other configuration from args
        maxEpochs = self.args.max_epochs
        initialLearningRate = self.args.initial_lr
        learningRateDecay = self.args.lr_decay
        momentum = self.args.momentum
        interactionVertex = self.args.interaction_vertex
        initialTemperature = self.args.initial_temperature
        finalTemperature = self.args.final_temperature
        temperatureRamp = self.args.temperature_ramp
        errorPower = self.args.error_power
        precision = self.args.precision
        task = self.args.task

        # The neurons to train, either masked by the function call or all neurons
        neuronMask = neuronMask if neuronMask is not None else torch.arange(self.dimension)
        # The size of neuron-wise batches. If not passed, use all neurons in one batch
        neuronBatchSize = self.neuronBatchSize if self.neuronBatchSize is not None else self.dimension
        # Calculate the number of neuron batches. Note this will smooth the number of neurons in each batch,
        # so the passed neuronBatchSize is more of an upper limit
        numNeuronBatches = np.ceil(neuronMask.shape[0] / neuronBatchSize).astype(int)
        # Get the neuron batches, sets of indices to train at once
        neuronBatches = torch.chunk(neuronMask, numNeuronBatches)

        # A tensor of all item indices
        itemIndices = torch.arange(X.shape[1])
        # The size of the item-wise batches. If not passed, use all items in one batch
        itemBatchSize = self.itemBatchSize if self.itemBatchSize is not None else X.shape[1]
        # Calculate the number of item batches. Note this will smooth the number of items in each batch,
        # so the passed itemBatchSize is more of an upper limit
        numItemBatches = np.ceil(itemIndices.shape[0] / itemBatchSize).astype(int)

        history_train = []
        history_test = []
        latest_test_acc = None  # Variable to store the most recent test accuracy

        memoryGrads = torch.zeros_like(self.memories).to(self.memories.device)
        epochProgressbar = tqdm(range(maxEpochs), desc="Epoch", disable=(verbose!=1))
        for epoch in range(maxEpochs):
            epochTotalLoss = 0
            totalCount = 0
            totalCorrect = 0 # accuracy

            # Shuffle the learned items so we are not learning the exact same batches each epoch
            shuffledIndices = torch.randperm(X.shape[1])
            X = X[:, shuffledIndices]
            
            learningRate = initialLearningRate*learningRateDecay**epoch
            if epoch < temperatureRamp:
                temperature = initialTemperature + (finalTemperature-initialTemperature) * epoch/temperatureRamp
            else:
                temperature = finalTemperature

            # Determine the value of beta for this epoch
            beta = 1/(temperature**interactionVertex) # before : 1/(temperature)

            # Determine the batches for this epoch, based on the newly shuffled states and the previously calculated batch numbers
            itemBatches = torch.chunk(X, numItemBatches, dim=1)
            
            for itemBatchIndex in range(numItemBatches):
                itemBatch = itemBatches[itemBatchIndex].detach()
                currentItemBatchSize = itemBatch.shape[1]

                itemBatchLoss = 0
                for neuronBatchIndex in range(numNeuronBatches):
                    
                    neuronIndices = neuronBatches[neuronBatchIndex].detach()
                    neuronBatchNumIndices = neuronIndices.shape[0]
                
                    tiledBatchClampOn = torch.tile(itemBatch, (1,neuronBatchNumIndices))
                    tiledBatchClampOff = torch.clone(tiledBatchClampOn)
                    for i, d in enumerate(neuronIndices):
                        tiledBatchClampOn[d,i*currentItemBatchSize:(i+1)*currentItemBatchSize] = 1
                        tiledBatchClampOff[d,i*currentItemBatchSize:(i+1)*currentItemBatchSize] = -1
                    
                    onSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOn)
                    offSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOff)
                    Y = torch.tanh(beta * torch.sum(onSimilarity-offSimilarity, axis=0)).reshape([neuronBatchNumIndices, currentItemBatchSize])
                    
                    ''' before :
                    onSimilarity = self.interactionFunction((beta / self.dimension) * (self.memories.T @ tiledBatchClampOn))
                    offSimilarity = self.interactionFunction((beta / self.dimension) * (self.memories.T @ tiledBatchClampOff))
                    Y = torch.tanh(torch.sum(onSimilarity-offSimilarity, axis=0)).reshape([neuronBatchNumIndices, currentItemBatchSize])
                    '''
                    
                    neuronBatchLoss = torch.sum((Y - itemBatch[neuronIndices])**(2*errorPower))
                    itemBatchLoss += neuronBatchLoss
                    
                    
                    # Calculate metrics based on task type
                    totalCount += Y.shape[1]
                    if task == "classification":
                        # Accumulate correct predictions by comparing argmax of predicted and target vectors
                        totalCorrect += (torch.argmax(Y, axis=0) == torch.argmax(itemBatch[neuronIndices], axis=0)).sum().item()
                
                itemBatchLoss.backward()
                with torch.no_grad():
                    epochGrads = self.memories.grad
                    memoryGrads = momentum * memoryGrads + epochGrads
                    maxGradMagnitude = torch.max(torch.abs(memoryGrads), axis=0).values.reshape(1, self.memories.shape[1])
                    maxGradMagnitude[maxGradMagnitude<precision] = precision
                    maxGradMagnitudeTiled = torch.tile(maxGradMagnitude, (self.dimension, 1))
                    self.memories -= learningRate * memoryGrads / maxGradMagnitudeTiled
                    if clampMemories: self.memories = self.memories.clamp_(-1,1)
                    self.memories.grad = None
                    epochTotalLoss += itemBatchLoss.item() / (neuronMask.shape[0] * itemIndices.shape[0])

            # After each epoch's updates, save the current state of memories
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils import saveStatesAsImage

            saveStatesAsImage(
                self.memories,
                numImages=num_display_memories,
                log_paths=log_paths,
                filename=f"memories",
                imageShape=image_shape,
                fig_kw={"figsize": (12,12)},
                title=f"Learned Memories - Epoch {epoch}"
            )

            # Store training epoch results with epoch number
            epoch_results = {
                "epoch": epoch,
                "loss": epochTotalLoss
            }
            if task == "classification":
                train_accuracy = (totalCorrect / totalCount) * 100
                epoch_results["accuracy"] = train_accuracy

            history_train.append(epoch_results)

            # Evaluate on test set if provided and it's evaluation interval or last epoch
            if X_test is not None and (epoch % eval_interval == 0 or epoch == maxEpochs - 1):
                test_results = self.evaluate(X_test, neuronMask=neuronMask, beta=beta, task=task)
                test_results["epoch"] = epoch
                latest_test_acc = test_results.get("accuracy")
                history_test.append(test_results)

                if verbose == 2:
                    log_msg = f"Epoch {epoch:04}: Loss {epoch_results['loss']:.4e}, Train Loss {epoch_results['loss']:.4e}"
                    if task == "classification":
                        log_msg += f", Train Accuracy {train_accuracy:.2f}%, Test Accuracy {test_results['accuracy']:.2f}%"
                    print(log_msg)
            
            # Progress bar update with latest metrics
            if verbose == 1:
                postfix = {"Loss": f"{epoch_results['loss']:4e}"}
                if task == "classification":
                    postfix["Train Acc"] = f"{train_accuracy:.2f}%"
                    if latest_test_acc is not None:
                        postfix["Latest Test Acc"] = f"{latest_test_acc:.2f}%"
                epochProgressbar.set_postfix(postfix)
                epochProgressbar.update()
            
            # Update plots
            for log_path in log_paths:
                plots_dir = os.path.join(log_path, "plots")
                os.makedirs(plots_dir, exist_ok=True)
            
            # Extract data from history
            epochs = [h["epoch"] for h in history_train]
            train_losses = [h["loss"] for h in history_train]
            
            # Plot training loss
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, 'b-', label='Training Loss')
            if history_test:  # If we have test data
                test_epochs = [h["epoch"] for h in history_test]
                test_losses = [h["loss"] for h in history_test]
                plt.plot(test_epochs, test_losses, 'r-', label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs')
            plt.legend()
            plt.grid(True)
            for log_path in log_paths:
                plt.savefig(os.path.join(log_path, 'plots/loss.png'))
            plt.close()
            
            if task == "classification":
                train_accuracies = [h["accuracy"] for h in history_train]
                
                # Plot accuracies
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
                if history_test:
                    test_accuracies = [h["accuracy"] for h in history_test]
                    plt.plot(test_epochs, test_accuracies, 'r-', label='Test Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title('Accuracy over Epochs')
                plt.legend()
                plt.grid(True)
                for log_path in log_paths:
                    plt.savefig(os.path.join(log_path, 'plots/accuracy.png'))
                plt.close()
                
            if log_paths:
                for path in log_paths:
                    csv_dir = os.path.join(path, "csv_data")
                    os.makedirs(csv_dir, exist_ok=True)
                    
                    # Save training history to CSV
                    train_df = pd.DataFrame(history_train)
                    train_csv_path = os.path.join(csv_dir, "train_history.csv")
                    train_df.to_csv(train_csv_path, index=False)
                    
                    # Save test history to CSV if it exists
                    if history_test:
                        test_df = pd.DataFrame(history_test)
                        test_csv_path = os.path.join(csv_dir, "test_history.csv")
                        test_df.to_csv(test_csv_path, index=False)

                    if verbose >= 2:
                        print(f"Training history saved to: {train_csv_path}")
                        if history_test:
                            print(f"Test history saved to: {test_csv_path}")

        return history_train, history_test
    
    @torch.no_grad()
    def energy(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates and returns the energy of a set of states.
        Energy is calculated as the interaction function applied to the sum of all memories dot the state.
        -F(sum M.T @ x)

        :param X: States, a tensor of shape (network.dimension, nStates).
        :return: A tensor of shape (nStates) measuring the energy of each state.
        """

        return -self.interactionFunction(torch.sum(self.memories.T @ X, axis=0))
    
    # def stable(self, X: torch.Tensor):
    #     """
    #     Calculate the stability of each state given.

    #     :param X: States, a tensor of shape (network.dimension, nStates).
    #     :returns: A (nStates) tensor of booleans with each entry the stability of a state.
    #     """

    @torch.no_grad()
    def stepStates(self, X: torch.Tensor, neuronMask: torch.Tensor = None, activationFunction: Callable = utils.BipolarHeaviside, scalingFactor: float = 1.0):
        """
        Step the given states according to the energy difference rule. 
        Step implies only a single update is made, no matter if the result is stable or not.

        Note X must have shape (network.dimension, nStates) where n is the number of states to update
        X must already be moved to the correct device. This can be done with X.to(network.device)

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (network.dimension, nStates)
        :param neuronMask: A mask of neuron indices to update. If passed, only the specified indices are updated. Other indices will be clamped.
            If None (default), all indices will be updated.
        :param activationFunction: The function to apply to the resulting step. For complex activation functions, use
            currying via lambda (e.g. `lambda X: torch.nn.Softmax(dim=0)(X)`)
        :param scalingFactor: A scaling factor to multiply similarity measure by before passing into the activation function.
            This can improve performance for interaction functions without inverses, such as RePOLY. 
            If your interaction function has an inverse, this should have no effect and can be left as 1.0
        """

        neuronMask = neuronMask if neuronMask is not None else torch.arange(self.dimension)
        neuronBatchSize = self.neuronBatchSize if self.neuronBatchSize is not None else self.dimension
        numNeuronBatches = np.ceil(neuronMask.shape[0] / neuronBatchSize).astype(int)
        neuronIndexBatches = torch.chunk(neuronMask, numNeuronBatches)

        itemIndices = torch.arange(X.shape[1])
        itemBatchSize = self.itemBatchSize if self.itemBatchSize is not None else X.shape[1]
        numItemBatches = np.ceil(itemIndices.shape[0] / itemBatchSize).astype(int)
        itemIndexBatches = torch.chunk(itemIndices, numItemBatches)

        for itemBatchIndices in itemIndexBatches:
            itemBatchIndices = itemBatchIndices.detach()
            currentItemBatchSize = itemBatchIndices.shape[0]
            items = X[:, itemBatchIndices]

            for neuronBatchIndices in neuronIndexBatches:
                neuronBatchIndices = neuronBatchIndices.detach()
                neuronBatchNumIndices = neuronBatchIndices.shape[0]

                tiledBatchClampOn = torch.tile(items, (1,neuronBatchNumIndices))
                tiledBatchClampOff = torch.clone(tiledBatchClampOn)
                for i, d in enumerate(neuronBatchIndices):
                    tiledBatchClampOn[d,i*currentItemBatchSize:(i+1)*currentItemBatchSize] = 1
                    tiledBatchClampOff[d,i*currentItemBatchSize:(i+1)*currentItemBatchSize] = -1
                
                raise Exception("wrong implementation on beta")
            
                onSimilarity = self.interactionFunction((scalingFactor / self.dimension) * (self.memories.T @ tiledBatchClampOn))
                offSimilarity = self.interactionFunction((scalingFactor / self.dimension) * (self.memories.T @ tiledBatchClampOff))
                
                Y = activationFunction(torch.sum(onSimilarity-offSimilarity, axis=0)).reshape([neuronBatchNumIndices, currentItemBatchSize])
                X[neuronBatchIndices[:, None], itemBatchIndices] = Y

    @torch.no_grad()
    def relaxStates(self, X: torch.Tensor, maxIterations: int = 100, verbose: bool = False, **stepStates_kwargs):
        """
        Update the states some number of times.

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (network.dimension, nStates)
        :param maxIterations: The integer number of iterations to update the states for.
        :param verbose: Flag to show progress bar
        :param stepStates_kwargs: Any additional keywords to pass directly to the stepStates call each epoch.
        """
        
        for _ in tqdm(range(maxIterations), desc="Relax States", disable=not verbose):
            X_prev = X.clone()
            self.stepStates(X, **stepStates_kwargs)
            if torch.all(X_prev == X):
                break
        
    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, neuronMask: torch.Tensor = None, beta: float = None, task: str = None) -> dict:
        """
        Evaluate the model on the given dataset.

        :param X: Test dataset tensor of shape (network.dimension, nStates)
        :param neuronMask: A mask of neuron indices to consider. If None, all neurons are used.
        :param beta: The beta value to use for the interaction function. If None, the beta value from the last epoch is used.
        :param task: Type of task ('classification' or None)
        :return: Dictionary containing evaluation metrics (loss and accuracy for classification)
        """
        
        beta = 1/(self.args.final_temperature**self.args.interaction_vertex) if beta is None else beta 
        
        neuronMask = neuronMask if neuronMask is not None else torch.arange(self.dimension)
        neuronBatchSize = self.neuronBatchSize if self.neuronBatchSize is not None else self.dimension
        numNeuronBatches = np.ceil(neuronMask.shape[0] / neuronBatchSize).astype(int)
        neuronBatches = torch.chunk(neuronMask, numNeuronBatches)

        itemIndices = torch.arange(X.shape[1])
        itemBatchSize = self.itemBatchSize if self.itemBatchSize is not None else X.shape[1]
        numItemBatches = np.ceil(itemIndices.shape[0] / itemBatchSize).astype(int)
        itemBatches = torch.chunk(X, numItemBatches, dim=1)

        totalLoss = 0
        totalCorrect = 0
        totalCount = 0

        for itemBatch in itemBatches:
            currentItemBatchSize = itemBatch.shape[1]

            for neuronBatchIndex in range(numNeuronBatches):
                neuronIndices = neuronBatches[neuronBatchIndex].detach()
                neuronBatchNumIndices = neuronIndices.shape[0]

                tiledBatchClampOn = torch.tile(itemBatch, (1, neuronBatchNumIndices))
                tiledBatchClampOff = torch.clone(tiledBatchClampOn)
                for i, d in enumerate(neuronIndices):
                    tiledBatchClampOn[d, i*currentItemBatchSize:(i+1)*currentItemBatchSize] = 1
                    tiledBatchClampOff[d, i*currentItemBatchSize:(i+1)*currentItemBatchSize] = -1
                
                onSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOn)
                offSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOff)
                Y = torch.tanh(beta * torch.sum(onSimilarity-offSimilarity, axis=0)).reshape([neuronBatchNumIndices, currentItemBatchSize])
                    
                # Calculate loss
                batchLoss = torch.sum((Y - itemBatch[neuronIndices])**2)
                totalLoss += batchLoss.item()

                if task == "classification":
                    totalCount += Y.shape[1]
                    totalCorrect += (torch.argmax(Y, axis=0) == torch.argmax(itemBatch[neuronIndices], axis=0)).sum().item()

        # Normalize loss
        avgLoss = totalLoss / (neuronMask.shape[0] * itemIndices.shape[0])
        
        results = {"loss": avgLoss}
        if task == "classification":
            accuracy = (totalCorrect / totalCount) * 100
            results["accuracy"] = accuracy
        
        return results

        