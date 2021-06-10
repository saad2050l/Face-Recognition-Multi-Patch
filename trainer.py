import torch
import numpy as np
import itertools
from itertools import count

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, nb_patch, margin, metrics=[], 
        start_epoch=0):
        
    for epoch in range(0, start_epoch):
        scheduler.step()
    tr_loss, vl_loss, acc_train, acc_val = [], [], [], []
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, nb_patch, margin)
        tr_loss.append(train_loss)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            acc_train.append(metric.value())
    
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics, margin)
        vl_loss.append(val_loss)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            acc_val.append(metric.value())

        print(message)
    return tr_loss, vl_loss, acc_train, acc_val


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, nb_patch, margin):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    
    for batch_idx, (data0, target0), (data1, target1), (data2, target2), (data3, target3), (data4, target4), (data5, target5), (data6, target6)  in zip(count(), train_loader[0], train_loader[1], train_loader[2], train_loader[3], train_loader[4], train_loader[5], train_loader[6]):
        tab = [(data0, target0), (data1, target1), (data2, target2), (data3, target3), (data4, target4), (data5, target5), (data6, target6)]
        for (data, target) in tab :
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                target = target.cuda()

        target0 = target0.cuda()
        optimizer.zero_grad()
        outputs = model(*data0, *data1, *data2, *data3, *data4, *data5, *data6)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs

        loss_outputs = loss_fn(*loss_inputs, target0)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # On s'en tape pour le moment mais ÄÂ  faire attention
        for metric in metrics:
            metric(outputs, target0, margin)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data0[0]), len(train_loader[0].dataset),
                100. * batch_idx / len(train_loader[0]), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    #torch.save(model.state_dict(), "paramĂ¨tre_model" + str(nb_patch))
    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics, margin):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        
        for batch_idx, (data0, target0), (data1, target1), (data2, target2), (data3, target3), (data4, target4), (data5, target5), (data6, target6)  in zip(count(), val_loader[0], val_loader[1], val_loader[2], val_loader[3], val_loader[4], val_loader[5], val_loader[6]):
        
            tab = [(data0, target0), (data1, target1), (data2, target2), (data3, target3), (data4, target4), (data5, target5), (data6, target6)]
            for (data, target) in tab :
                if not type(data) in (tuple, list):
                    data = (data,)
                if cuda:
                    data = tuple(d.cuda() for d in data)
                    target = target.cuda()
            
            target0 = target0.cuda()
            outputs = model(*data0, *data1, *data2, *data3, *data4, *data5, *data6)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs

            loss_outputs = loss_fn(*loss_inputs, target0)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target0, margin)

    return val_loss, metrics
