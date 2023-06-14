import dtlpy as dl
import logging
import os
from omegaconf import OmegaConf as om
from scripts.train import train

dl.client_api.verbose.print_error_logs = True
dl.client_api.cookie_io.put('cache_mode', 'dict')

logger = logging.getLogger(name='mpt')
logger.setLevel(logging.INFO)


@dl.Package.decorators.module(name="mpt-trainer",
                              description="wrapper to run mpt training as a FaaS")
class ServiceRunner(dl.BaseServiceRunner):

    @dl.Package.decorators.function(display_name="Train MPT",
                                    inputs={
                                        "dataset": "Dataset",
                                        "data_folder": "String"
                                    },
                                    outputs={})
    def train(self, dataset: dl.Dataset, data_folder: str = "my-copy-c4"):
        logger.info(f"We are currently at dir: {os.getcwd()}")
        logger.info(f"Current state: {os.listdir()}")
        data_folder_exists = os.path.exists(data_folder)
        output_folder_exists = os.path.exists("./mpt-125m")
        if data_folder_exists:
            os.rmdir(data_folder)
        os.makedirs(f"./{data_folder}")
        logger.info(f"Created dir {data_folder}: {os.listdir()}")
        os.makedirs(f"./{data_folder}/train_small")
        logger.info(f"Created dir {data_folder}/train_small: {os.listdir(data_folder)}")
        os.makedirs(f"./{data_folder}/val_small")
        logger.info(f"Created dir {data_folder}/val_small: {os.listdir(data_folder)}")

        def download_folder(folder_type: str):
            folder_filter = dl.Filters(field="filename", values=f"/{data_folder}/{folder_type}_small/**")
            for page in dataset.items.list(filters=folder_filter):
                for item in page:
                    file = item.download(local_path=f"./{data_folder}/{folder_type}_small/{item.name}")
            logger.info(f"{folder_type} files downloaded: {os.listdir(f'./{data_folder}/{folder_type}_small')}")

        stages = [x for x in ['train', 'val'] if len(os.listdir(f"./{data_folder}/{x}_small")) == 0]
        for t in stages:
            download_folder(t)

        yaml_path = "./125m.yaml"
        if not os.path.exists(yaml_path):
            yaml_file = dataset.items.get("/125m.yaml").download(local_path=yaml_path)
            logger.info(f"Config file 125m.yaml downloaded: {os.listdir()}")
        else:
            logger.info("Config already existed.")

        with open(yaml_path) as f:
            yaml_cfg = om.load(f)

        if not output_folder_exists:
            os.makedirs("./mpt-125m")
            logger.info(f"Created output directory: {os.listdir()}")
        else:
            logger.info("Output dir already existed")

        args_list = [f'data_local={data_folder}',
                     'train_loader.dataset.split=train_small',
                     'eval_loader.dataset.split=val_small',
                     'max_duration=10ba',
                     'eval_interval=0',
                     'save_folder=./mpt-125m']
        cli_cfg = om.from_cli(args_list)
        cfg = om.merge(yaml_cfg, cli_cfg)
        train.main(cfg)

        for it in os.listdir("./mpt-125m"):
            dataset.items.upload(local_path=os.path.join("./mpt-125m", it),
                                 remote_path="./mpt-125m",
                                 remote_name=it)

    @dl.Package.decorators.function(display_name="Get Dataset",
                                    inputs={
                                        "item": "Item"
                                    },
                                    outputs={
                                        "dataset": "Dataset"
                                    })
    def get_dataset(self, item: dl.Item) -> dl.Dataset:
        return item.dataset
