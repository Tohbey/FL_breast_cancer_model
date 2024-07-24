import flwr as fl
from typing import List, Tuple
from utils import plotClientData

results_list = []

def aggregate(results: List[Tuple[int, List]]):
    # Initialize the aggregated weights with the shape of the model weights
    aggregated_weights = [weights * 0 for weights in results[0][1]]
    
    # Sum the weights from all clients
    for client_weights in results:
        for i, layer_weights in enumerate(client_weights[1]):
            aggregated_weights[i] += layer_weights

    # Compute the average of the weights
    num_clients = len(results)
    aggregated_weights = [layer_weights / num_clients for layer_weights in aggregated_weights]
    
    return aggregated_weights

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[int, float]],
        failures: List[BaseException]
    ) -> float:
        if not results:
            return float('inf')

        # Compute weighted average
        loss_aggregated = sum([num_examples * loss for num_examples, loss in results]) / sum([num_examples for num_examples, _ in results])

        # Store the results
        accuracy_aggregated = sum([accuracy for _, accuracy in results]) / len(results)
        results_list.append({"round": rnd, "loss": loss_aggregated, "accuracy": accuracy_aggregated})
        
        print(f"Round {rnd} - Loss: {loss_aggregated}, Accuracy: {accuracy_aggregated}")
        
        return loss_aggregated

# Define the custom strategy
strategy = CustomFedAvg(aggregate_fn=aggregate)

# Start the Flower server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 10},
        strategy=strategy
    )

    # Print the results list after training is complete
    print("Training results:")
    plotClientData(results_list)
