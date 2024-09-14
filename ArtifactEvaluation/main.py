import argparse

# miscellaneous
import builtins
import datetime
import sys
import time
import ctypes
import os
import os.path as osp

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
from torch.autograd.profiler import record_function
from torch.utils.tensorboard import SummaryWriter

# datasets
from load_data import make_datasets_and_loaders

# models
from models import DLRM_Net, WDL_Net, DCN_Net

# quotient-remainder trick
from tricks.md_embedding_bag import md_solver
from tricks.sk_embedding_bag import get_sketch_time
from tricks.sk_embedding_bag import reset_sketch_time

exc = getattr(builtins, "IOError", "FileNotFoundError")


def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(dlrm, X, lS_o, lS_i, use_gpu, device, test=False, sk_flag=False):
    with record_function("DLRM forward"):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            if sk_flag:
                if X != None:
                    return dlrm(X.to(device), lS_o, lS_i, test)
                else:
                    return dlrm(None, lS_o, lS_i, test)
            else:
                lS_i = (
                    [S_i.to(device) for S_i in lS_i]
                    if isinstance(lS_i, list)
                    else lS_i.to(device)
                )
                lS_o = (
                    [S_o.to(device) for S_o in lS_o]
                    if isinstance(lS_o, list)
                    else lS_o.to(device)
                )
        if X != None:
            return dlrm(X.to(device), lS_o, lS_i, test)
        else:
            return dlrm(None, lS_o, lS_i, test)


# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


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
    test=False,
    sk_flag=False,
):
    test_accu = 0
    test_samp = 0
    scores = []
    targets = []
    
    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break
        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )
        # forward pass
        Z_test = dlrm_wrap(
            dlrm,
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            test=test,
            sk_flag=sk_flag,
        )
        if Z_test.is_cuda:
            torch.cuda.synchronize()

        with record_function("DLRM accuracy compute"):
            # compute loss and accuracy

            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            scores.append(S_test)
            targets.append(T_test)

            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum(
                (np.round(S_test, 0) == T_test).astype(np.uint8))

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


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Train recommendation model.")
    # model
    parser.add_argument("--model", type=str, default='dlrm', choices=['dlrm', 'wdl', 'dcn'])
    # model related parameters
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument(
        "--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument(
        "--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=10)
    parser.add_argument("--lr-flag", action="store_true", default=False)
    parser.add_argument("--hash-flag", action="store_true", default=False)
    parser.add_argument("--bucket-flag", action="store_true", default=False)
    parser.add_argument("--sketch-flag", action="store_true", default=False)
    parser.add_argument("--compress-rate", type=float, default=0.001)
    parser.add_argument("--hc-threshold", type=int, default=200)
    parser.add_argument("--hash-rate", type=float, default=0.5)

    # data
    parser.add_argument("--data-set", type=str, default="criteo",
                        choices=['criteo', 'criteotb', 'avazu', 'kdd12'])
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=8)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time",
                        action="store_true", default=False)
    parser.add_argument("--enable-profiling",
                        action="store_true", default=False)
    parser.add_argument("--tensor-board-filename",
                        type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")

    parser.add_argument("--notinsert-test", action="store_true", default=False)
    parser.add_argument("--ada-flag", action="store_true", default=False)

    parser.add_argument("--data_path", type=str, required=True, help="data dir path")

    parser.add_argument("--sketch-threshold", type=int, default=500)

    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

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

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    train_data, train_ld, test_data, test_ld = make_datasets_and_loaders(args)
    nbatches = len(train_ld)
    nbatches_test = len(test_ld)
    embedding_nums = train_data.counts
    print(embedding_nums)
    hash_rate = 0
    # enforce maximum limit on number of vectors per embedding
    if args.max_ind_range > 0:
        embedding_nums = np.minimum(embedding_nums, args.max_ind_range)
    m_den = ln_bot[0] = train_data.num_dense
    has_dense = m_den > 0

    ### parse command line arguments ###
    embedding_dim = args.embedding_dim
    hotn = 0
    if args.sketch_flag:
        totn = sum(embedding_nums)
        hotn = int(totn * args.compress_rate * (1 - args.hash_rate)
                   * (embedding_dim * 4 / (embedding_dim * 4 + 48)))
        hash_rate = args.compress_rate * args.hash_rate
        print(f"hash_rate: {hash_rate}, hotn: {hotn}")
    embedding_nums = np.asarray(embedding_nums)
    num_fea = embedding_nums.size + (m_den > 0)
    if m_den == 0:
        m_den_out = 0
    else:
        m_den_out = ln_bot[ln_bot.size - 1]
    # approach 1: all
    # num_int = num_fea * num_fea + m_den_out
    # approach 2: unique
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    print(arch_mlp_top_adjusted)
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    if args.model == 'dlrm':
        if args.qr_flag and args.data_set != 'avazu' and args.data_set != 'kdd12':
            if args.qr_operation == "concat" and 2 * embedding_dim != m_den_out:
                sys.exit(
                    "ERROR: 2 embedding_dim "
                    + str(2 * embedding_dim)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                    + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
                )
            if args.qr_operation != "concat" and embedding_dim != m_den_out:
                sys.exit(
                    "ERROR: embedding_dim "
                    + str(embedding_dim)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                )
        else:
            if embedding_dim != m_den_out and args.data_set != 'avazu' and args.data_set != 'kdd12':
                sys.exit(
                    "ERROR: embedding_dim "
                    + str(embedding_dim)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                )
        if num_int != ln_top[0]:
            sys.exit(
                "ERROR: # of feature interactions "
                + str(num_int)
                + " does not match first dimension of top mlp "
                + str(ln_top[0])
            )

    # assign mixed dimensions if applicable
    if args.md_flag:
        l = 0.0001
        r = 0.5
        while r - l > 0.0001:
            mid = (l + r) / 2
            embedding_dim_ = md_solver(
                torch.tensor(embedding_nums),
                mid,  # alpha
                d0=embedding_dim,
                round_dim=args.md_round_dims,
            ).tolist()
            cr = sum(embedding_dim_ * embedding_nums) / (np.sum(embedding_nums) * embedding_dim)
            if cr > args.compress_rate:
                l = mid
            else:
                r = mid
        embedding_dim = md_solver(
            torch.tensor(embedding_nums),
            r,  # alpha
            d0=embedding_dim,
            round_dim=args.md_round_dims,
        ).tolist()

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    lib = None
    if args.sketch_flag:
        trick_dir = osp.join(osp.split(osp.abspath(__file__))[0], 'tricks')
        os.system(f"g++ -fPIC -shared -o {trick_dir}/sklib.so -g -rdynamic -mavx2 -mbmi -mavx512bw -mavx512dq --std=c++17 -O3 -fopenmp {trick_dir}/sketch.cpp")
        lib = ctypes.CDLL(f'{trick_dir}/sklib.so')
    # init model
    model_cls = {
        'dlrm': DLRM_Net,
        'wdl': WDL_Net,
        'dcn': DCN_Net,
    }[args.model]
    dlrm = model_cls(
        args.compress_rate,
        args.ada_flag,
        lib,
        hotn,
        device,
        args.sketch_flag,
        hash_rate,
        args.sketch_threshold,
        embedding_dim,
        embedding_nums,
        ln_bot,
        ln_top,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
    )

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()

    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            # "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = (
            dlrm.parameters()
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)

    ### main loop ###

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
        if args.sketch_flag:
            dlrm.lib.update()
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

    tb_file = "./" + args.tensor_board_filename
    writer = SummaryWriter(tb_file)

    with torch.autograd.profiler.profile(
        args.enable_profiling, use_cuda=use_gpu, record_shapes=True
    ) as prof:
        if not args.inference_only:
            k = 0
            total_time_begin = 0
            while k < args.nepochs:

                if k < skip_upto_epoch:
                    continue

                num = 0
                t1 = 0
                t2 = 0
                t3 = 0
                for j, inputBatch in enumerate(train_ld):

                    if j < skip_upto_batch:
                        continue
                    X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
                    t3 = t1
                    t1 = time_wrap(use_gpu)

                    # early exit if nbatches was set by the user and has been exceeded
                    if nbatches > 0 and j >= nbatches:
                        break

                    # = args.mini_batch_size except maybe for last
                    mbs = T.shape[0]

                    # forward pass
                    Z = dlrm_wrap(
                        dlrm,
                        X,
                        lS_o,
                        lS_i,
                        use_gpu,
                        device,
                        test=False,
                        sk_flag=args.sketch_flag,
                    )

                    # loss
                    with record_function("DLRM loss compute"):
                        E = dlrm.loss_fn(Z, T.to(device))

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    # training accuracy is not disabled
                    # S = Z.detach().cpu().numpy()  # numpy array
                    # T = T.detach().cpu().numpy()  # numpy array

                    # # print("res: ", S)

                    # # print("j, train: BCE ", j, L)

                    # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    # A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                    with record_function("DLRM backward"):
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        optimizer.zero_grad()
                        # backward pass
                        E.backward()
                        # grad_num = 0
                        # grad_offset = 0
                        # for name, parms in dlrm.named_parameters():
                        #     print('-->name:', name)
                        #     print('-->para:', torch.max(parms), torch.min(parms))
                        #     #print('-->grad_requirs:',parms.requires_grad)
                        #     print('-->grad_value:', parms.grad.shape, parms.grad, parms.grad.indices)
                        #     grad_num += 1
                        #     if grad_num == 26:
                        #         break

                        if args.sketch_flag:
                            dlrm.insert_grad(lS_i)
                        if args.ada_flag:
                            dlrm.insert_adagrad(lS_i)
                        # optimizer
                        optimizer.step()

                    total_time += t1 - t3

                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    should_print = ((j + 1) % args.print_freq == 0) or (
                        j + 1 == nbatches
                    ) or (j <= 100)
                    should_test = (
                        (args.test_freq > 0)
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if should_print or should_test:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        train_loss = total_loss / total_samp
                        total_loss = 0

                        str_run_type = (
                            "inference" if args.inference_only else "training"
                        )

                        wall_time = ""
                        if args.print_wall_time:
                            wall_time = " ({})".format(time.strftime("%H:%M"))

                        print(
                            "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                                str_run_type, j + 1, nbatches, k, gT,
                            )
                            + " loss {:.6f}".format(train_loss)
                            + wall_time,
                            flush=True,
                        )

                        log_iter = nbatches * k + j + 1
                        writer.add_scalar("Train/Loss", train_loss, log_iter)

                        total_iter = 0
                        total_samp = 0
                        reset_sketch_time()

                    # testing

                    if should_test:
                        epoch_num_float = (j + 1) / len(train_ld) + k + 1
                        # dlrm.grad_norm = np.load("grad_norm.npy")
                        # print(f"sum = {sum(dlrm.grad_norm)}")

                        print(
                            "Testing at - {}/{} of epoch {},".format(
                                j + 1, nbatches, k)
                        )
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
                            args.notinsert_test,
                            args.sketch_flag,
                        )

                        if (
                            is_best
                            and not (args.save_model == "")
                            and not args.inference_only
                        ):
                            model_metrics_dict["epoch"] = k
                            model_metrics_dict["iter"] = j + 1
                            model_metrics_dict["train_loss"] = train_loss
                            model_metrics_dict["total_loss"] = total_loss
                            model_metrics_dict[
                                "opt_state_dict"
                            ] = optimizer.state_dict()
                            print("Saving model to {}".format(args.save_model))
                            torch.save(model_metrics_dict, args.save_model)

                        # Uncomment the line below to print out the total time with overhead
                        # print("Total test time for this group: {}" \
                        # .format(time_wrap(use_gpu) - accum_test_time_begin))

                k += 1  # nepochs
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

    # profiling
    if args.enable_profiling:
        time_stamp = str(datetime.datetime.now()).replace(" ", "_")
        with open("dlrm_s_pytorch" + time_stamp + "_shape.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cpu_time_total"
                )
            )
        with open("dlrm_s_pytorch" + time_stamp + "_total.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(
                sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("dlrm_s_pytorch" + time_stamp + ".json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    total_time_end = time_wrap(use_gpu)


if __name__ == "__main__":
    run()
