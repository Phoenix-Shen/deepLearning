from models import FedAvg
import yaml
if __name__=="__main__":
    with open("./FederatedLearning/FedAvg/config.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    print(config)
    
    model = FedAvg(config)
    model.train()
    