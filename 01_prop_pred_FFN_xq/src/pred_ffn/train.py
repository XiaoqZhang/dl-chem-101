"""
    Train ffn to predict targets. 
"""

# There should be a train_model() function

# import pkgs
import argparse
import yaml
import logging
from pathlib import Path
from tdc.single_pred import ADME

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from pred_ffn import ffn_data, ffn_model, utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    # devices
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--devices", default=8, action="store", type=int)
    parser.add_argument("--num_workers", default=0, action="store", type=int)
    # task settings
    parser.add_argument("--dataset_name", default="caco", choices=["caco"])
    parser.add_argument("--save_dir", default="results/example_run")
    parser.add_argument("--batch_size", default=64, action="store", type=int)
    parser.add_argument("--max_epochs", default=100, action="store", type=int)
    # model parameters
    parser.add_argument("--layers", default=3, action="store", type=int)
    parser.add_argument("--lr", default=1e-3, action="store", type=float)
    parser.add_argument("--dropout", default=0.1, action="store", type=float)
    parser.add_argument("hidden-size", default=128, action="store", type=int)

    return parser.parse_args()

def train_model():
    # get arguments 
    args = get_args()
    kwargs = args.__dict__
    seed_everything(kwargs["seed"])

    # save arguments
    save_dir = kwargs["save_dir"]
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info(f"Args: \n{yaml_args}")

    # Extract data
    if kwargs["dataset_name"] == "caco":
        data = ADME(name = 'Caco2_Wang')
        split = data.get_split()
        train_smi, train_y = zip(*split["train"][["Drug", "Y"]].values)
        valid_smi, valid_y = zip(*split["valid"]["Drug", "Y"].values)
        test_smi, test_y = zip(*split["test"]["Drug", "Y"].values)
    else:
        raise NotImplementedError()

    # Prepare dataset
    num_workers = kwargs["num_workers"]
    train_dataset = ffn_data.PredDataset(
        train_smi,
        train_y,
        num_workers=num_workers
    )
    valid_dataset = ffn_data.PredDataset(
        valid_smi,
        valid_y,
        num_workers=num_workers
    )
    test_dataset = ffn_data.PredDataset(
        test_smi,
        test_y,
        num_workers=num_workers
    )
    dataset_sizes = (len(train_dataset), len(valid_dataset), len(test_dataset))
    logging.info(f"Train, valid, test dataset sizes: {dataset_sizes}. ")

    # Define dataloaders
    batch_size = kwargs["batch_size"]
    collect_fn = train_dataset.get_collect_fn()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collect_fn=collect_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collect_fn=collect_fn   # Why does here use the same collect_fn from train_dataset?
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collect_fn=collect_fn
    )

    # Define model 
    model = ffn_model.ForwardFFN(
        layers = kwargs["layers"],
        hidden_size = kwargs["hidden_size"],
        learning_rate = kwargs["lr"],
        dropout = kwargs["dropout"],
        output_dim = 1
    )

    # Create trainer
    ## loggers
    db_logger = pl_loggers.WandbLogger(save_dir=save_dir) # to be defined
    console_logger = utils.ConsoleLogger() # to be defined
    
    ## callbacks 
    db_path = db_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=db_path,
        filename="{epoch}-{val_loss:.2f}",
        auto_insert_metric_name=False
    ) 
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=10)
    callbacks = [earlystop_callback, checkpoint_callback]
    
    ## trainer
    trainer = Trainer(
        accelerator="gpu" if kwargs["gpu"] else "cpu",
        # auto_lr_find
        callbacks = callbacks,
        # check_val_every_n_epoch
        devices = 1 if kwargs["gpu"] else kwargs["devices"],
        gpus = 1 if kwargs["gpu"] else 0,
        # log_every_n_steps
        gradient_clip_val = 5, # a good initialization is the average value of gradient norms
        gradient_clip_algorithm = "value"   # gradient clipping to avoid graident boosting, another option is "norm"
        logger = [db_logger, console_logger],
        max_epoches = kwargs["max_epochs"],
        enable_process_bar = True,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)

    # Load best checkpoint
    best_checkpoint = checkpoint_callback.best_model_path
    best_checkpoint_score = checkpoint_callback.best_model_score.item()
    best_model = ffn_model.ForwardFFN.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model from {best_checkpoint} with val loss of {best_checkpoint_score}. ")

    best_model.eval()
    test_out = trainer.test(dataloaders=test_dataloader)

    out_yaml = {"args:" kwargs, "test_metrics": test_out[0]}
    out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "test_results.yaml", "w") as fp:
        fp.write(out_str)
