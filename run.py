"""
Usage:
    run.py test --step=<int>
    run.py train
    run.py val --step=<int>
    run.py tune

Options:
    -h --help                               show this screen.
    --step=<int>                            which step of parameter to be loaded
"""
from dataprocessor import FileProcessor
from dataprocessor import DataProcessor
from rnn import MoleculeRNNTrainer
from docopt import docopt
from rnn import HyperParameterTuner


class Config:
    split_ratio = 0.9
    num_epoch = 1
    hidden_size = 12
    clip_grad = 0.5
    lr = 0.15
    dropout_ratio = 0.6
    log_every = 1
    batch_size = 256
    test_sample = 500000
    use_attention = True
    turn_parameter = {"hidden_size": [6, 12], "dropout_ratio": [0.5, 0.8], "attention": [True, False],
                      "lr": [0.1, 0.5], "batch_size": [128, 256]}

# tensorboard --logdir C:\Kaggle\Molecule\log --host=127.0.0.1
# python -m torch.utils.bottleneck run.py train
# python "C:\Users\szzha\Anaconda3\Scripts\kernprof-script.py" -l -v run.py train


if __name__ == '__main__':

    args = docopt(__doc__)

    # files = FileProcessor()
    # train, test = files.get_stucture()
    # data_processor = DataProcessor(train, test)
    # data_processor.add_distance().cat_encode().filter().save_dataset()
    data_processor = DataProcessor()
    data_processor.load_dataset(Config.test_sample)
    runner = MoleculeRNNTrainer(data_processor, Config.split_ratio, Config.num_epoch, Config.hidden_size,
                                 Config.clip_grad, Config.lr, Config.dropout_ratio, Config.log_every, Config.batch_size,
                                use_attention=Config.use_attention)
    if args['test']:
        runner.load_model(num_iter=args["--step"])
        runner.test()
    elif args['val']: # debug purpose, should never use purely
        runner.load_model(num_iter=args["--step"])
        runner.set_val_each_step(False)
        runner.validate()
    elif args['train']:
        # runner.load_model(num_iter=1700)
        runner.set_val_each_step(False)
        runner.train()
        # runner.set_val_each_step(False)
        # runner.validate()
    elif args['tune']:
        runner.set_val_each_step(True)
        tuner = HyperParameterTuner(Config.turn_parameter, runner)
        tuner.tune()
