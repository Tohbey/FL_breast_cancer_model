import flwr as fl
from typing import List, Tuple

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

# Define a simple strategy for federated learning
strategy = fl.server.strategy.FedAvg(aggregate_fn=aggregate)

# Start the Flower server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 5},
        strategy=strategy
    )