import argparse, random, glob, os, shutil
from model import *
from util import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='mtref', type=str,
                    choices=['mtref', 'wiki', 'newsela', 'arxiv', 'msr', 'edinburgh'], required=True)
parser.add_argument('--out', default='../out/', type=str)
parser.add_argument('--unsupervised_dir', default='../out/unsupervised_final/', type=str)
parser.add_argument('--sure_and_possible', action='store_true')
parser.add_argument('--model', default='bert-base-cased', type=str)
parser.add_argument('--trained_model', default=None, type=str)
parser.add_argument('--ot_type', help='OT type', type=str, choices=['ot', 'pot', 'uot', 'none'],
                    required=True)
parser.add_argument('--weight_type', help='Weight type', type=str, default='--',
                    choices=['norm', 'uniform', 'predict', '--'])
parser.add_argument('--dist_type', help='Distance metric', type=str, default='--', choices=['cos', 'l2', '--'])
parser.add_argument('--div_type', help='uot_mm divergence', type=str, default='--', choices=['kl', 'l2', '--'])
parser.add_argument('--attention', action='store_true')
parser.add_argument('--lr', help='learning rate', type=float, default=3e-5)
parser.add_argument('--batch', default=1, type=int)
parser.add_argument('--batch_span', default=1, type=int)
parser.add_argument('--train_max_word_num', default=512, type=int)
parser.add_argument('--train_max_epochs', default=200, type=int)
parser.add_argument('--limit_batch_ratio', default=1.0, type=float)
parser.add_argument('--patience', default=5, type=int)
parser.add_argument('--seed', help='number of attention heads', type=int, default=42)
parser.add_argument('--save_model', action='store_true', help='Save model')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
gpus = torch.cuda.device_count()


if __name__ == '__main__':
    Aligner = NeuralAligner
    if args.trained_model is not None:
        trained_model_path = glob.glob(os.path.join(args.trained_model, '**/', '*.ckpt'), recursive=True)
        assert len(trained_model_path) == 1
        trained_model_path = trained_model_path[0]
        model = Aligner.load_from_checkpoint(trained_model_path, data=args.data,
                                             sure_and_possible=args.sure_and_possible)

        save_dir = os.path.dirname(trained_model_path)
        logger = TensorBoardLogger(save_dir=args.trained_model, name=args.data)
        trainer = pl.Trainer(gpus=gpus, logger=logger)
        trainer.test(model)

    else:
        batch_size = args.batch
        accum_grad_batch = int(128 / batch_size) if args.data == 'wiki' else 1
        setting_yaml = 'lr_setting.yml'
        lr = load_lr_setting(setting_yaml, args.patience, args.sure_and_possible, args.ot_type)
        unsupervised_result_dir = args.unsupervised_dir + '{0}_sure-possible-{1}/{2}_{3}_{4}/{5}/'.format(args.data,
                                                                                                          args.sure_and_possible,
                                                                                                          args.ot_type,
                                                                                                          args.weight_type,
                                                                                                          args.dist_type,
                                                                                                          args.seed)
        ot_hyp = load_ot_setting(unsupervised_result_dir, args.ot_type)
        distortion = load_distortion_setting('distortion_setting.yml', args.data, args.sure_and_possible)

        save_dir = '{0}/{1}_sure-possible-{2}/{3}_{4}_{5}/{6}/'.format(args.model.replace('/', '_'),
                                                                           args.data,
                                                                           args.sure_and_possible,
                                                                           args.ot_type,
                                                                           args.weight_type, args.dist_type, args.seed)

        logger = TensorBoardLogger(save_dir=args.out, name=save_dir)

        # swa_callback = StochasticWeightAveraging(swa_epoch_start=3)
        lr_monitor = LearningRateMonitor(logging_interval='step')

        model = Aligner(args.model, args.data, args.sure_and_possible,
                        batch_size, lr, ot_type=args.ot_type, weight_type=args.weight_type, dist_type=args.dist_type,
                        div_type=args.div_type, attention=args.attention, ot_hyp=ot_hyp, distortion=distortion,
                        train_max_word_num=args.train_max_word_num,
                        batch_span=args.batch_span)

        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint_callback = ModelCheckpoint(
            monitor="val_total_f1",
            filename="neuralopt_{epoch:02d}-{val_total_f1:.6f}",
            save_top_k=1,
            mode="max",
        )
        # Early stopping callback
        early_stop_callback = EarlyStopping(
            monitor='val_total_f1',
            min_delta=1e-4,
            patience=args.patience,
            verbose=False,
            mode='max'
        )

        # w/o learning rate tuning
        trainer = pl.Trainer(gpus=gpus, logger=logger, check_val_every_n_epoch=1,
                             log_every_n_steps=max(accum_grad_batch * 5, 10), max_epochs=args.train_max_epochs,
                             limit_train_batches=args.limit_batch_ratio, limit_val_batches=args.limit_batch_ratio,
                             limit_test_batches=args.limit_batch_ratio,
                             accumulate_grad_batches=accum_grad_batch,
                             callbacks=[checkpoint_callback, early_stop_callback, lr_monitor])

        trainer.fit(model)

        # automatically loads the best weights for you
        trainer.test(ckpt_path="best")
        if args.data == 'mtref':  # Test Newsela and ArXiv (w/o training dataset)
            model.data = 'newsela'
            trainer.test(ckpt_path="best")
            model.data = 'arxiv'
            trainer.test(ckpt_path="best")

        if not args.save_model:
            shutil.rmtree(trainer.logger.log_dir + '/checkpoints/')
