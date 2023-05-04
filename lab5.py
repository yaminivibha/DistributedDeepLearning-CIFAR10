"""Benchmark CIFAR10 with PyTorch on CPU & GPU"""
# Code Attribution: https://github.com/kuangliu/pytorch-cifar
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from prettytable import PrettyTable

from models import *
from utils import load_data, print_config, progress_bar, set_optimizer

EXERCISES = ["Q1"]

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument("exercise", default="C1", help="problem # on HW")
    parser.add_argument("--outfile", default="outfile.txt", help="output filename")
    parser.add_argument("--epochs", default=2, type=int, help="num epochs; default 2")
    parser.add_argument("--optimizer", default="SGD", help="optimizer, default SGD")
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=2,
        help="dataloader workers; default 2",
    )
    parser.add_argument(
        "--data_path", default="./data", help="data dirpath; default ./data"
    )
    parser.add_argument(
        "--lr", default=0.1, type=float, help="learning rate, default 0.1"
    )
    parser.add_argument(
        "--cuda", default=True, action="store_true", help="cuda usage; default True"
    )
    parser.add_argument("--batch_size", default=128, type=int, help="batch size; default 128")
    parser.add_argument("--gpus", default=1, type=int, help="num gpus; default 1")
    parser.add_argument(
        "--no_batch_norms",
        default=False,
        action="store_true",
        help="no batch norms; default False",
    )
    args = parser.parse_args()

    # Config
    print("==> Setting configs..")
    if args.exercise not in EXERCISES:
        raise ValueError("Invalid exercise")
    args.device = "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    args.optimizer = set_optimizer(args)
    args.filename = args.outfile if args.outfile else args.exercise + ".txt"
    print_config(args)
    outfile = open(args.filename, "a")

    # Data
    print("==> Preparing data..")
    trainloader, trainset, testloader, testset = load_data(args)

    # Model Setup
    print("==> Building model..")
    if args.no_batch_norms:
        net = ResNet18NoBatchNorm()
    else:
        net = ResNet18()
    net = net.to(args.device)
    if args.device == "cuda":
        if(args.gpus == 1):
            net = torch.nn.DataParallel(net)
        elif(args.gpus == 2):
            net == torch.nn.DataParallel(net, [0,1])
        elif(args.gpus == 3):
            net == torch.nn.DataParallel(net, [0,1,2])
        elif(args.gpus == 4):
            net == torch.nn.DataParallel(net, [0,1,2,3])
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = args.optimizer(net.parameters(), **args.hyperparameters)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training
    def train(epoch):
        """
        Execute one epoch of training.
        Args:

            epoch (int): the current epoch
        Returns:
            c2_load_time (float): the time spent loading data
        """
        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        sum_train_loss = 0

        c2_load_time = 0
        c2_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            c2_load_time += time.time() - c2_start
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(trainloader),
                "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
            sum_train_loss += train_loss
            c2_start = time.time()
        ave_train_loss = sum_train_loss / len(trainloader)
        return {"load_time": c2_load_time, "ave_train_loss": ave_train_loss}

    def test(epoch):
        """
        Tests the model on the test set.
        Args:
            epoch (int): the current epoch
        Returns:
            c2_load_time (float): the time spent loading data
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            c2_load_time = 0
            c2_start = time.time()
            for batch_idx, (inputs, targets) in enumerate(testloader):
                c2_load_time += time.time() - c2_start
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Test Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
                c2_start = time.time()

            return {"load_time": c2_load_time, "accuracy": 100 * correct / total}

    #### LAB5 Q1 ####
    if args.exercise == "Q1":
        print(f"======== LAB5 ========", file=outfile)
        print_config(args)

        args.device = "cuda"
        print("==> Preparing data..")
        trainloader, trainset, testloader, testset = load_data(args)
        print("==> Synchronizing GPU..")
        torch.cuda.synchronize()  # wait for warm-up to finish

        train_times = []
        accuracies = []
        average_train_losses = []
        for epoch in range(args.epochs):
            torch.cuda.synchronize()  # wait for warm-up to finish

            start_time = time.time()
            loss = train(epoch)["ave_train_loss"]
            train_time = time.time()
            scheduler.step()

            average_train_losses.append(loss)
            train_times.append(train_time - start_time)
            accuracy = test(epoch)["accuracy"]
            accuracies.append(accuracy)

            print(f"Epoch {epoch} ", file=outfile)
            print(f"    Train Time {train_time - start_time}\n", file=outfile)
            print(f"    Accuracy {accuracy}\n", file=outfile)

        print(
            f"#### Q1 Summary For Batch Size = {args.batch_size} ####\n\n",
            file=outfile,
        )
        table = PrettyTable([])
        table.add_column("Epoch", [1,])
        table.add_column("Training Time (secs)", train_times[1])
        table.add_column("Accuracy", accuracies[1])
        table.add_column("Average Train Loss", average_train_losses[1])
        print(table, file=outfile)
        outfile.close()
        return
        


if __name__ == "__main__":
    main()
