import flwr as flower


def aggregate(results):
    weights = [r[1] for r in results]
    return weights

# Define a simple strategy for federated learning
strategy = flower.server.strategy.FedAvg(aggregate_fn=aggregate)

# Start the Flower server
flower.server.start_server(
    server_address="0.0.0.0:8080",
    config={"num_rounds": 5}, 
    strategy=strategy
)