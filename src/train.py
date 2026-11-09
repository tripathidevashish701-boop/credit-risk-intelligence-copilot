from .model import train_model

if __name__ == "__main__":
    result = train_model()
    print("Evaluation metrics")
    for key, value in result.metrics.items():
        print(f"{key}: {value}")
