import argparse
from basic.data import ScoreDataset
from basic.Static import Statistics
from basic.NumEmbedding import NumEmbedding
import torch
import torch.optim as optim
import os
import random
import numpy as np
import logging.config


os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
# device = torch.device("cuda:0")

seed_num = 57
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def main():
    args = parse_args()
    # make dir
    args.data_path = args.data_path + args.data_name + "/nums.pt"
    args.model_path = args.model_path + args.data_name + "/" + args.model_name + "/"
    args.output_path = args.output_path + args.data_name + "/" + args.model_name + "/"
    args.log_path = args.log_path + args.data_name + "/"
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    args.log_path = args.log_path + args.model_name + ".log"
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logging.info(args)

    # read data
    data_dict = torch.load(args.data_path)
    train_data = data_dict['train']
    dev_data = data_dict['dev']
    test_data = data_dict['test']

    logging.info('Num Data loaded...')

    # train batch
    train_dataset = ScoreDataset(train_data, 'train', args.batch_size, args.pad_ind, args.gpu)
    dev_dataset = ScoreDataset(dev_data, 'dev', args.batch_size, args.pad_ind, args.gpu)
    test_dataset = ScoreDataset(test_data, 'test', args.batch_size, args.pad_ind, args.gpu)

    # model
    model = NumEmbedding(args)
    if args.gpu:
        model = model.cuda()

    # checkpoint = torch.load(
    #     '/home/chenshaowei/AAAI2021/model/rotowire/2_0.005lr_0trunc/20.pth')
    # model.load_state_dict(checkpoint['net'])
    # batch_generator_test = test_dataset.get_batch_data()
    # test(model, batch_generator_test, args.beam_size, args.min_length, args.max_length,
    #      vocab_dict['tgt'], args.output_path + str(28) + "_beam_2.txt", args.gpu)
    # exit(0)

    # optimizer
    param_list = list(model.named_parameters())
    # for n, p in param_list:
    #     if p.requires_grad == True:
    #         print(n)
    grouped_params = [p for n, p in param_list if p.requires_grad == True]

    # make optimizer
    if args.optim_type == "adam":
        optimizer = optim.Adam(grouped_params, lr=args.learning_rate, betas=[0.9, 0.999], eps=1e-09)
    elif args.optim_type == "sgd":
        optimizer = optim.SGD(grouped_params, lr=args.learning_rate)
    elif args.optim_type == "adagrad":
        optimizer = optim.Adagrad(grouped_params, lr=args.learning_rate,
                                  initial_accumulator_value=args.adagrad_accum)
    elif args.optim_type == "adadelta":
        optimizer = optim.Adadelta(grouped_params, lr=args.learning_rate)
    elif args.optim_type == "rmsprop":
        optimizer = optim.RMSprop(grouped_params, lr=args.learning_rate)
    else:
        print('Error Optimizer!')

    # load saved model, optimizer and epoch num
    if args.reload and os.path.exists(args.model_path + '{}.pth'.format(args.saved_epoch_num)):
        checkpoint = torch.load(args.model_path + '{}.pth'.format(args.saved_epoch_num))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('Reload model and optimizer after training epoch {}'.format(checkpoint['epoch']))
    else:
        start_epoch = 1
        print('New model and optimizer from epoch 0')

    best_dev_ppl = 0.
    last_ppl = None
    ppl = None
    lr = args.learning_rate
    for epoch in range(start_epoch, start_epoch + args.epoch_num):
        model.train()

        # updata learning rate
        start_decay = False
        if args.start_decay_at is not None and epoch >= args.start_decay_at:
            start_decay = True
        if last_ppl is not None and ppl is not None:
            if ppl > last_ppl:
                start_decay = True

        if start_decay:
            lr = lr * args.learning_rate_decay
            optimizer.param_groups[0]["lr"] = lr
            logging.info("Decaying learning rate to %g" % lr)
        if ppl is not None:
            last_ppl = ppl

        batch_generator = train_dataset.get_batch_data()

        total_stats = Statistics()
        report_stats = Statistics()
        for id, (step, batch_num, src, tgt, src_lengths, src_mask) in enumerate(batch_generator):

            report_stats.n_src_words += src_lengths.sum().item()
            total_stats.n_src_words += src_lengths.sum().item()

            optimizer.zero_grad()

            decoder_outputs, attns, dec_state = model(src, tgt_inp_temp, src_lengths, dec_state)
            loss, batch_stats = model.compute_loss(decoder_outputs, attns, tgt_oup_temp, src_map,
                                                   align_temp, copy_vocab)
            loss.backward()  # no normalization
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(grouped_params, args.max_grad_norm)
            optimizer.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

            # log
            if step % 100 == 0:
                report_stats.output(epoch, step, batch_num)
                report_stats = Statistics()
            # if step / 100 == 3:
            #     state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            #     torch.save(state, args.model_path + str(epoch) + '.pth')
            #     batch_generator_test = test_dataset.get_batch_data()
            #
            #     test_greedy(model, batch_generator_test, args.beam_size, args.min_length, args.max_length,
            #          vocab_dict['tgt'], args.output_path + str(epoch) + ".txt", args.gpu)

        # validation
        batch_generator_dev = dev_dataset.get_batch_data()
        dev_state = valid(model, batch_generator_dev)
        dev_state.output_dev(epoch)
        # logging.info(dev_state.ppl())

        # save checkpoint
        if epoch > args.start_checkpoint_at:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, args.model_path + str(epoch) + '.pth')

        # test
        batch_generator_test = test_dataset.get_batch_data()
        test(model, batch_generator_test, args.beam_size, args.min_length, args.max_length,
            vocab_dict['tgt'], args.output_path+str(epoch)+".txt", args.gpu)


def parse_args():
    # config
    parser = argparse.ArgumentParser(description='numberic embedding')
    parser.add_argument('--model_name', type=str, default="initial_pretrain_num_emb")
    parser.add_argument('--data_name', type=str, default="rotowire")
    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--model_path', type=str, default="model/")
    parser.add_argument('--output_path', type=str, default="output/")
    parser.add_argument('--log_path', type=str, default="log/")

    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--pad_ind', type=int, default=0)
    parser.add_argument('--src_vocab_size', type=list, default=[])
    parser.add_argument('--tgt_vocab_size', type=int, default=1000)
    parser.add_argument('--emb_size', type=int, default=600)
    parser.add_argument('--emb_out_size', type=int, default=600)
    parser.add_argument('--tgt_emb_dim', type=int, default=500)
    parser.add_argument('--encoder_dim', type=int, default=600)

    parser.add_argument('--epoch_num', type=int, default=40)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--optim_type', type=str, default="adam",
                        choices=["adam", "sgd", "adagrad", "adadelta", "rmsprop"])
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--adagrad_accum', type=float, default=0.15)
    parser.add_argument('--learning_rate_decay', type=float, default=0.98)
    parser.add_argument('--max_grad_norm', type=float, default=5)

    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--decoder_dim', type=int, default=600)
    parser.add_argument('--coverage_attn', type=bool, default=True)
    parser.add_argument('--attn_type', type=str, default="mlp", choices=["dot", "general", "mlp"])

    parser.add_argument('--reuse_copy_attn', type=bool, default=True)

    parser.add_argument('--reload', type=bool, default=False)
    parser.add_argument('--saved_epoch_num', type=int, default=0)
    parser.add_argument('--start_decay_at', type=int, default=15)
    parser.add_argument('--start_checkpoint_at', type=int, default=0)

    parser.add_argument('--gen_length', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=400)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()