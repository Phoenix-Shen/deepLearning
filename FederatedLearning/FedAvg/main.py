from models import FedAvg
import yaml
import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file",type=str,help="your config file path",default="./FederatedLearning/FedAvg/configs/config.yaml")
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    print(config)
    
    model = FedAvg(config)
    model.train()
    