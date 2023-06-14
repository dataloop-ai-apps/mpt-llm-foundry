import json
import logging
import os
import torch
import shutil
import warnings
import dtlpy as dl
from glob import glob
from omegaconf import OmegaConf as om
from scripts.train import train
from scripts.inference import convert_composer_to_hf, hf_generate
from scripts.data_prep import convert_dataset_json, convert_dataset_hf, convert_finetuning_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from argparse import Namespace

logger = logging.Logger("MPT-ADAPTER")


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for MPT LLM',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    model = None
    tokenizer = None

    def save(self, local_path: str, **kwargs):
        self.model_entity.artifacts.upload(os.path.join(local_path, '*'))
        self.configuration.update({'model_filename': 'weights/latest.pt'})

    def load(self, local_path: str, **kwargs):
        trained_model_path = os.path.join(local_path, 'trained_model')
        if os.path.exists(trained_model_path) and "config.json" in os.listdir(trained_model_path):
            load_config = self.configuration.get("load", {})
            # Grab config first
            print(f'Loading HF Config...')
            from_pretrained_kwargs = {
                'use_auth_token': load_config.get("use_auth_token", False),
                'trust_remote_code': load_config.get("trust_remote_code"),
                'revision': load_config.get("revision")
            }
            try:
                hf_config = AutoConfig.from_pretrained(trained_model_path,
                                                       **from_pretrained_kwargs)
                if load_config.get("attn_impl") is not None and hasattr(hf_config, 'attn_config'):
                    hf_config.attn_config['attn_impl'] = load_config.get("attn_impl")
                if load_config.get("max_seq_len") is not None and hasattr(hf_config, 'max_seq_len'):
                    hf_config.max_seq_len = load_config.get("max_seq_len")
            except Exception as e:
                raise RuntimeError(
                    'If you are having auth problems, try logging in via `huggingface-cli login` '
                    'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... '
                    'using your access token from https://huggingface.co/settings/tokens.'
                ) from e

            def get_dtype(dtype: str) -> torch.dtype:
                if dtype == 'fp32':
                    return torch.float32
                elif dtype == 'fp16':
                    return torch.float16
                elif dtype == 'bf16':
                    return torch.bfloat16
                else:
                    raise NotImplementedError(
                        f'dtype {dtype} is not supported. '
                        f'We only support fp32, fp16, and bf16 currently')

            # Set device and model_dtype
            if self.configuration.get("device") is not None:
                device = self.configuration.get("device")
            else:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            if load_config.get("model_dtype") is not None:
                model_dtype = get_dtype(load_config.get("model_dtype"))
            else:
                model_dtype = hf_config.torch_dtype or torch.float32

            # Load HF Model
            print(f'Loading HF model to device={device} and dtype={model_dtype}...')
            try:
                self.model = AutoModelForCausalLM.from_pretrained(trained_model_path,
                                                                  config=hf_config,
                                                                  torch_dtype=model_dtype,
                                                                  **from_pretrained_kwargs)
                self.model.to(device)
                print(f'n_params={sum(p.numel() for p in self.model.parameters())}')
            except Exception as e:
                raise RuntimeError(
                    'If you are having auth problems, try logging in via `huggingface-cli login` '
                    'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... '
                    'using your access token from https://huggingface.co/settings/tokens.'
                ) from e

            print('\nLoading HF tokenizer...')
            self.tokenizer = AutoTokenizer.from_pretrained(trained_model_path,
                                                           **from_pretrained_kwargs)
            if self.tokenizer.pad_token_id is None:
                warnings.warn(
                    'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        else:
            logger.error("Failed to load model!")

    def train(self, data_path, output_path, **kwargs):
        if hasattr(self.model, "train"):
            self.model.train()
        # Initializing train configuration:
        train_cfg = self.configuration.get("train", {})
        train_cfg["train_loader.dataset.split"] = train_cfg["train_loader_dataset_split"]
        train_cfg["eval_loader.dataset.split"] = train_cfg["eval_loader_dataset_split"]
        train_cfg["train_loader"]["dataset"]["split"] = None if train_cfg["train_loader"]["dataset"]["split"] == ''\
            else train_cfg["train_loader"]["dataset"]["split"]
        train_cfg['train_loader']['dataset']['hf_kwargs'] = {'data_dir': data_path}
        train_cfg = om.create(train_cfg)
        train_cfg.data_local = self.configuration.get("convert_data", {}).get("out_root", data_path)
        train_cfg.save_folder = output_path
        train.main(train_cfg, self.model, self.tokenizer)
        self.model_entity.dataset.items.upload(os.path.join(output_path, train_cfg.save_latest_filename),
                                               remote_path="/.dataloop")
        # Initializing converter configuration. The converter takes torch model and converts to HF format
        convert_cfg = self.configuration.get("convert", {})
        convert_cfg["hf_output_path"] = os.path.join(output_path, "trained_model")
        convert_cfg["hf_repo_for_upload"] = None
        convert_cfg["local_checkpoint_save_location"] = None
        convert_cfg["composer_path"] = os.path.join(output_path, train_cfg.save_latest_filename)
        if convert_cfg.get("is_mpt_model", True):
            convert_composer_to_hf.main(Namespace(**convert_cfg))
        else:
            convert_composer_to_hf.write_huggingface_pretrained_from_composer_checkpoint(
                checkpoint_path=convert_cfg["composer_path"],
                output_path=convert_cfg["hf_output_path"],
                output_precision=convert_cfg.get("output_precision", "fp32")
                )
        self.load_from_model(self.model_entity, output_path)
        if not self.model:
            raise FileNotFoundError("Model trained, but its files weren't found. Trained model was lost.")

    def predict(self, batch, **kwargs):
        generate_config = self.configuration.get("generate", {})
        generate_config["loaded"] = self.model is not None
        if generate_config["loaded"]:
            generate_config["model"] = self.model
            generate_config["tokenizer"] = self.tokenizer
        else:
            raise ValueError("Model not loaded, cannot predict at the moment.")
        if hasattr(self.model, "eval"):
            self.model.eval()
        generate_config['prompts'] = [prompt for prompt in batch]
        generate_config['max_seq_len'] = None
        generate_config['temperature'] = 1.0
        generate_config['top_k'] = 50
        generate_config['top_p'] = 1.0
        generate_config['do_sample'] = True
        generate_config['use_cache'] = True
        generate_config['model_dtype'] = self.configuration.get("load_config", {}).get("model_dtype", "fp32")
        generate_config['seed'] = 1337
        generate_config['attn_impl'] = None
        generate_config['device'] = None
        generate_config['revision'] = None
        generate_config['use_auth_token'] = False
        generate_config['trust_remote_code'] = True
        generate_config['warmup'] = True
        generate_config['autocast_dtype'] = None
        generate_config['eos_token_id'] = None
        generate_config['pad_token_id'] = None
        generate_config["device"] = self.configuration.get("device", "cpu")
        generate_config['name_or_path'] = os.path.join(os.getcwd(), "output", "trained_model")
        outputs = hf_generate.main(Namespace(**generate_config))
        timestamp = datetime.now().strftime(f"%H-%M-%S_%d-%m-%Y")
        tags = []
        for i, (prompt, response) in enumerate(zip(batch, outputs)):
            text_annotation = dl.AnnotationCollection()
            print(f"[Prompt]: {prompt}")
            print("#" * 30)
            print(f"[Reponse]: {response}")
            txt = f"[Prompt]: {prompt}\n" + "#" * 100 + f"\n[Response]: {response}"
            with open(f"output_{i}.txt", "w") as f:
                f.write(txt)
            self.model_entity.dataset.items.upload(f"output_{i}.txt", remote_path=f"/responses/{timestamp}")
            label = f"/responses/{timestamp}/output_{i}.txt"
            text_annotation.add(annotation_definition=dl.Classification(label),
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0})
            tags.append(text_annotation)
        return tags

    def convert_from_dtlpy(self, data_path, **kwargs):
        convert_configs = self.configuration.get("convert_data", {})
        if convert_configs["dataset_type"] != "finetune":
            converter_selector = {
                'hf': convert_dataset_hf,
                'json': convert_dataset_json,
                'finetune': convert_finetuning_dataset
            }
            path_key = "dataset" if convert_configs.get("dataset_type", 'json') == 'finetune' else 'path'
            convert_configs[path_key] = data_path
            convert_configs['compression'] = None
            if "splits" in convert_configs:
                convert_configs["splits"] = convert_configs["splits"].split(" ") \
                if isinstance(convert_configs["splits"], str) else convert_configs["splits"]
            if os.path.exists(convert_configs.get("out_root")):
                shutil.rmtree(convert_configs.get("out_root"))
            convert_configs["local"] = convert_configs.get("out_root", "local-converted-data")
            converter = converter_selector[convert_configs.get("dataset_type", 'json')]
            converter.main(Namespace(**convert_configs))
            if not os.path.exists(os.path.join(convert_configs.get("out_root"), "train")):
                converted_files = os.listdir(convert_configs.get("out_root"))
                os.makedirs(os.path.join(convert_configs.get("out_root"), "train"))
                for file in converted_files:
                    shutil.move(os.path.join(convert_configs.get("out_root"), file),
                                os.path.join(convert_configs.get("out_root"), "train", file))


def package_creation(project: dl.Project, old_ver=None):
    with open("finetune-config.json", "r") as config_file:
        default_config = json.load(config_file)
    metadata = dl.Package.get_ml_metadata(cls=Adapter,
                                          input_type='txt',
                                          default_configuration=default_config,
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')
    if old_ver:
        reqs = old_ver.requirements
    else:
        reqs = []
    package = project.packages.push(package_name='mpt-adapter-finetuning',
                                    src_path=os.getcwd(),
                                    package_type='ml',
                                    modules=[modules],
                                    requirements=reqs,
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_M,
                                                                        runner_image='mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json(),
                                        'initParams': {'model_entity': None}
                                    },
                                    metadata=metadata)
    return package


def model_creation(package: dl.Package, dataset: dl.Dataset, config_file_path: str, model_name: str):
    with open(config_file_path, "r") as config_file:
        default_config = json.load(config_file)
    model = package.models.create(model_name=model_name,
                                  description='mpt large language model',
                                  tags=['llm', 'mpt'],
                                  dataset_id=dataset.id,
                                  input_type='text',
                                  status='created',
                                  scope='project',
                                  train_filter=dl.Filters(field='dir', values='/train'),
                                  validation_filter=dl.Filters(field='dir', values='/val'),
                                  configuration=default_config,
                                  project_id=package.project.id,
                                  labels=[]
                                  )
    return model


def main():
    PROJECT_NAME = "nlp-experiments"
    DATASET_NAME = "small-c4"
    project = dl.projects.get(PROJECT_NAME)
    dataset = project.datasets.get(DATASET_NAME)
    pkf = project.packages.get("mpt-adapter")
    pkf = package_creation(project, pkf)
    model = model_creation(pkf, dataset)

