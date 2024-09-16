import argparse

# miscellaneous
import time
import os.path as osp

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
from torch.utils.tensorboard import SummaryWriter

# datasets, embeddings, models
from load_data import make_datasets_and_loaders
from embeddings.init_embed import EmbeddingLayer
from models import DLRM_Net, WDL_Net, DCN_Net


def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
    device,
    use_gpu,
    nbatches,
    nbatches_test,
    writer,
    log_iter=-1,
):
    test_accu = 0
    test_samp = 0
    scores = []
    targets = []
    
    for it, testBatch in enumerate(test_ld):
        dense_test, offsets_test, indices_test, targets_test = testBatch
        if dense_test != None:
            dense_test = dense_test.to(device)
        preds_test = dlrm(dense_test, offsets_test, indices_test)
        preds_test = preds_test.detach().cpu().numpy()
        targets_test = targets_test.detach().cpu().numpy()
        scores.append(preds_test)
        targets.append(targets_test)

        mbs_test = targets_test.shape[0]
        A_test = np.sum(
            (np.round(preds_test, 0) == targets_test).astype(np.uint8))

        test_accu += A_test
        test_samp += mbs_test

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    metrics = {
        "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "ap": sklearn.metrics.average_precision_score,
        "roc_auc": sklearn.metrics.roc_auc_score,
    }

    validation_results = {}
    for metric_name, metric_function in metrics.items():
        validation_results[metric_name] = metric_function(targets, scores)
        writer.add_scalar(
            metric_name,
            validation_results[metric_name],
            log_iter,
        )

    acc_test = test_accu / test_samp
    writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    is_best = acc_test > best_acc_test
    if is_best:
        best_acc_test = acc_test
    print(
        " accuracy {:3.3f} %, auc {:3.3f} %, best {:3.3f} %".format(
            acc_test * 100,
            validation_results['roc_auc'] * 100,
            best_acc_test * 100
        ),
        flush=True,
    )
    return model_metrics_dict, is_best


def main():
    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Train recommendation model.")
    # model
    parser.add_argument("--model", type=str, default='dlrm',
                        choices=['dlrm', 'wdl', 'dcn'])
    # data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="criteo",
                        choices=['criteo', 'criteotb', 'avazu', 'kdd12'])
    # model related parameters
    parser.add_argument("--embedding_dim", type=int, required=True)
    # embedding table options
    parser.add_argument("--compress_method", type=str, default=None,
                        choices=[None, 'hash', 'mde', 'qr', 'ada', 'cafe'])
    parser.add_argument("--compress_rate", type=float, default=0.001)
    parser.add_argument("--max_ind_range", type=int, default=-1)

    # md flags
    parser.add_argument("--md_round_dims", type=bool, default=False)
    # cafe flags
    parser.add_argument("--cafe_sketch_threshold", type=int, default=500)
    parser.add_argument("--cafe_hash_rate", type=float, default=0.5)

    # gpu
    parser.add_argument("--use_gpu", type=bool, default=True)
    # training
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--print_precision", type=int, default=5)
    parser.add_argument("--numpy_rand_seed", type=int, default=123)
    parser.add_argument("--optimizer", type=str, default="sgd")
    # testing
    parser.add_argument("--inference_only", type=bool, default=False)
    parser.add_argument("--test_freq", type=int, default=-1)
    parser.add_argument("--test_mini_batch_size", type=int, default=16384)
    parser.add_argument("--test_num_workers", type=int, default=16)
    # debugging and profiling
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--print_time", type=bool, default=True)
    parser.add_argument("--print_wall_time", type=bool, default=False)
    parser.add_argument("--tensor_board_filename",
                        type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save_model", type=str, default="")
    parser.add_argument("--load_model", type=str, default="")

    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    use_gpu = args.use_gpu and torch.cuda.is_available()
    print(f"availble: {torch.cuda.is_available()}")

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        ngpus = torch.cuda.device_count()
        device = torch.device("cuda", 0)
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    # input data
    train_data, train_ld, test_data, test_ld = make_datasets_and_loaders(args)
    nbatches = len(train_ld)
    nbatches_test = len(test_ld)
    embedding_nums = train_data.counts
    print(embedding_nums)
    # enforce maximum limit on number of vectors per embedding
    if args.max_ind_range > 0:
        embedding_nums = np.minimum(embedding_nums, args.max_ind_range)
    m_den = train_data.num_dense

    ### prepare training data ###
    if args.dataset == 'criteotb':
        ln_bot = np.array([train_data.num_dense, 512, 256, args.embedding_dim], dtype=int)
    else:
        ln_bot = np.array([train_data.num_dense, 512, 256, 64, args.embedding_dim], dtype=int)

    ### parse command line arguments ###
    embedding_dim = args.embedding_dim
    embedding_nums = np.asarray(embedding_nums)
    num_fea = embedding_nums.size + (m_den > 0)
    if m_den == 0:
        m_den_out = 0
    else:
        m_den_out = ln_bot[-1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    if args.dataset == 'criteotb':
        ln_top = np.array([num_int, 1024, 1024, 512, 256, 1], dtype=int)
    else:
        ln_top = np.array([num_int, 512, 256, 1], dtype=int)

    # init embedding
    embedding_layer = EmbeddingLayer(args, embedding_dim, embedding_nums, device)

    # init model
    model_cls = {
        'dlrm': DLRM_Net,
        'wdl': WDL_Net,
        'dcn': DCN_Net,
    }[args.model]
    dlrm = model_cls(
        embedding_layer,
        device,
        embedding_dim,
        train_data.num_sparse,
        m_den,
        ln_bot,
        ln_top,
    )

    if use_gpu:
        dlrm = dlrm.to(device)

    if not args.inference_only:
        opts = {
            "sgd": torch.optim.SGD,
            "adagrad": torch.optim.Adagrad,
        }
        parameters = (
            dlrm.parameters()
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)

    # training or inference
    best_acc_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            # NOTE: when targeting inference on single GPU,
            # note that the call to .to(device) has already happened
            ld_model = torch.load(
                args.load_model,
                map_location=torch.device("cuda")
                # map_location=lambda storage, loc: storage.cuda(0)
            )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(
                args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])
        dlrm.embedding_layer.on_load()
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_acc_test = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}".format(
                ld_train_loss,
            )
        )

        print("Testing state: accuracy = {:3.3f} %".format(
            ld_acc_test * 100))

    print("time/loss/accuracy (if enabled):")

    tb_file = osp.join(osp.split(osp.abspath(__file__))[0], args.tensor_board_filename)
    writer = SummaryWriter(tb_file)

    if not args.inference_only:
        ep = 0
        while ep < args.nepochs:
            if ep < skip_upto_epoch:
                continue
            t1 = 0
            t2 = 0
            for it, batch in enumerate(train_ld):

                if it < skip_upto_batch:
                    continue
                dense, offsets, indices, targets = batch

                if args.print_time:
                    t2 = t1
                    if use_gpu:
                        torch.cuda.synchronize()
                    t1 = time.time()

                mbs = targets.shape[0]
                # forward pass
                if dense != None:
                    dense = dense.to(device)
                preds = dlrm(dense, offsets, indices)

                # loss
                loss = dlrm.loss_fn(preds, targets.to(device))

                # compute loss and accuracy
                loss_np = loss.detach().cpu().numpy()  # numpy array

                optimizer.zero_grad()
                loss.backward()
                dlrm.embedding_layer.insert_grad(indices)
                optimizer.step()

                if args.print_time:
                    total_time += t1 - t2
                total_loss += loss_np * mbs
                total_iter += 1
                total_samp += mbs

                should_print = ((it + 1) % args.print_freq == 0) or (
                    it + 1 == nbatches
                ) or (it <= 100)
                should_test = (
                    (args.test_freq > 0)
                    and (((it + 1) % args.test_freq == 0) or (it + 1 == nbatches))
                )

                # print time, loss and accuracy
                if should_print or should_test:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    train_loss = total_loss / total_samp
                    total_loss = 0

                    str_run_type = "inference" if args.inference_only else "training"

                    wall_time = ""
                    if args.print_wall_time:
                        wall_time = " ({})".format(time.strftime("%H:%M"))

                    print(
                        "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                            str_run_type, it + 1, nbatches, ep, gT,
                        )
                        + " loss {:.6f}".format(train_loss)
                        + wall_time,
                        flush=True,
                    )

                    log_iter = nbatches * ep + it + 1
                    writer.add_scalar("Train/Loss", train_loss, log_iter)

                    total_iter = 0
                    total_samp = 0

                # testing
                if should_test:
                    print(f"Testing at - {it + 1}/{nbatches} of epoch {ep},")
                    model_metrics_dict, is_best = inference(
                        args,
                        dlrm,
                        best_acc_test,
                        best_auc_test,
                        test_ld,
                        device,
                        use_gpu,
                        nbatches,
                        nbatches_test,
                        writer,
                        log_iter,
                    )

                    if (
                        is_best
                        and not (args.save_model == "")
                        and not args.inference_only
                    ):
                        model_metrics_dict["epoch"] = ep
                        model_metrics_dict["iter"] = it + 1
                        model_metrics_dict["train_loss"] = train_loss
                        model_metrics_dict["total_loss"] = total_loss
                        model_metrics_dict[
                            "opt_state_dict"
                        ] = optimizer.state_dict()
                        print("Saving model to {}".format(args.save_model))
                        torch.save(model_metrics_dict, args.save_model)

            ep += 1  # nepochs
    else:
        print("Testing for inference only")
        inference(
            args,
            dlrm,
            best_acc_test,
            best_auc_test,
            test_ld,
            device,
            use_gpu,
            nbatches,
            nbatches_test,
            writer,
        )


if __name__ == "__main__":
    main()
