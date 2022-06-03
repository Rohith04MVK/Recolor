from tqdm import tqdm

from .utils import (
    create_loss_meters, update_losses, log_results, visualize, AverageMeter
)


def train_model(model, train_dl, val_dl, epochs, display_every=200):
    """
    The function takes in a model, a training dataloader, a validation dataloader, the number of
    epochs to train for, and a display_every parameter which is the number of iterations after which the
    model's outputs are displayed
    
    :param model: the model object
    :param train_dl: the training dataloader
    :param val_dl: validation dataloader
    :param epochs: number of epochs to train for
    :param display_every: The number of iterations after which the model's output is displayed, defaults
    to 200 (optional)
    """
    data = next(iter(val_dl))  # getting a batch for visualizing the model output after fixed intrvals

    for e in range(epochs):
        loss_meter_dict = create_loss_meters()  # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network

        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()

            update_losses(model, loss_meter_dict, count=data['L'].size(0))  # function updating the log objects
            i += 1

            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")

                log_results(loss_meter_dict)  # function to print out the losses
                visualize(model, data, save=False)  # function displaying the model's outputs


def pretrain_generator(net_G, train_dl, opt, criterion, epochs, device):
    """
    It takes a generator network, a dataloader, an optimizer, a loss function, the number of epochs to
    train for, and a device to train on, and trains the generator network for the specified number of
    epochs
    
    :param net_G: the generator network
    :param train_dl: the training dataloader
    :param opt: the optimizer
    :param criterion: The loss function to use
    :param epochs: number of epochs to train for
    :param device: the device to run the training on
    """
    for e in range(epochs):
        loss_meter = AverageMeter()

        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")
