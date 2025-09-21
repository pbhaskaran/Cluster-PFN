import torch
import utils
device = torch.device("cuda")
import prior
import time
import transformer
import torch.nn as nn
import numpy as np
import argparse


def validate(model, criterion, X_val, y_val,targets_batch_clusters, val_mask):
    """Evaluate the model on validation data."""
    model.eval()
    with torch.no_grad():
        val_output,val_cluster_output = model(X_val, val_mask)
        val_cluster_output = val_cluster_output.view(-1, val_cluster_output.shape[2])
        targets_batch_clusters = targets_batch_clusters.reshape(-1).type(torch.LongTensor).to(device)
        targets_batch_clusters -= 1  ## super janky
        val_output = val_output.view(-1, val_output.shape[2])
        val_targets = y_val.reshape(-1).type(torch.LongTensor).to(device)
        val_loss = criterion(val_output, val_targets) +criterion(val_cluster_output, targets_batch_clusters)
    model.train()
    return val_loss

def train(model, criterion,start_epoch, num_epochs, optimizer, scheduler, batch_size,seq_len, num_features,cluster_type,check_point, **kwargs):
    model.train()
    trains = np.zeros(num_epochs - start_epoch)
    start_time = time.time()
    if cluster_type == 'bayesian_blobs':
        X_val, y_val, X_true, batch_clusters_val = prior.generate_bayesian_gmm_data(batch_size=2 * batch_size,
                                                                                    seq_len=seq_len,
                                                                                    num_features=num_features,seed=0,  **kwargs)
    else:
        X_val, y_val,X_true,batch_clusters_val = prior.sample_clusters(batch_size=2 * batch_size,seq_len=seq_len, num_features=num_features,cluster_type=cluster_type, **kwargs)

    val_mask = (torch.zeros(batch_clusters_val.shape)).long().to(device)

    for e in range(start_epoch, num_epochs):
        model.zero_grad()

        # Generate training data
        if cluster_type == 'bayesian_blobs':
            X, y, X_true, batch_clusters = prior.generate_bayesian_gmm_data(batch_size=batch_size, seq_len=seq_len,
                                                                            num_features=num_features, **kwargs)

        else:
            X, y, X_true, batch_clusters = prior.sample_clusters(batch_size=batch_size,seq_len=seq_len, num_features=num_features,cluster_type=cluster_type, **kwargs)
        mask = (torch.rand(batch_clusters.shape) > 0.5).long().to(device)
        batch_clusters_masked = mask * batch_clusters

        # forward pass
        output, batch_cluster_output  = model(X, batch_clusters_masked)
        targets_batch_clusters = batch_clusters

        # Reshape outputs and targets
        output = output.view(-1, output.shape[2])
        targets = y.reshape(-1).type(torch.LongTensor).to(device)
        batch_cluster_output = batch_cluster_output.view(-1, batch_cluster_output.shape[2])
        targets_batch_clusters = targets_batch_clusters.reshape(-1).type(torch.LongTensor).to(device)
        targets_batch_clusters -=1 ## super janky

        # compute loss
        loss_output = criterion(output, targets)
        loss_clusters = criterion(batch_cluster_output, targets_batch_clusters)

        loss = loss_output + loss_clusters

        # back prop
        loss.backward()
        optimizer.step()
        scheduler.step()


        val_loss = validate(model, criterion, X_val, y_val,batch_clusters_val, val_mask)
        trains[e - start_epoch] = val_loss.item()
        if e % 10000 == 0:
            print('| epoch {:3d} | lr {} || ''validation loss {:5.3f}'.format(
                e, scheduler.get_last_lr()[0], val_loss))
        if e > 0 and e % 10000 == 0:
            torch.save({
                'epoch':e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scheduler_last_epoch': scheduler.last_epoch,
            }, f"checkpoint_{check_point}_{kwargs.get('mean_precision_prior', 'default')}.pt")
            np.save(f"trains{start_epoch}.npy", trains)
    end_time = time.time()
    print(f"training completed in {end_time - start_time:.2f} seconds")
    return trains

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Transformer Clustering Model")
    print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--nhid', type=int, default=512)
    parser.add_argument('--nlayers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_outputs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--in_features', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=1000)
    parser.add_argument('--check_point', type=str, default="bayesian_blobs")
    parser.add_argument('--cluster_type', type=str, default='bayesian_blobs')
    parser.add_argument('--weight_concentration_prior', type=float, default=1.0)
    parser.add_argument('--mean_prior', type=float, default=0.0)
    parser.add_argument('--mean_precision_prior', type=float, default=0.01)
    parser.add_argument('--degrees_of_freedom_prior', type=float, default=None)
    parser.add_argument('--covariance_prior', type=float, default=None)
    parser.add_argument('--feature_attn', action='store_true')
    parser.add_argument("--nan_frac", type=float, default = 0.8)
    parser.add_argument('--model_path' , type=str, default=None)

    # covariance_prior = None
    args = parser.parse_args()

    vary_feature = False
    device = torch.device("cuda")
    model = transformer.Transformer(args.d_model, args.nhead, args.nhid, args.nlayers, in_features=args.in_features,
                                    buckets_size=args.num_outputs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warm_up_epochs = args.num_epochs * 0.05
    criterion = nn.CrossEntropyLoss()
    if args.model_path:
        print("loading pretrained model")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch' , 0)
        scheduler = utils.get_cosine_schedule_with_warmup(optimizer, warm_up_epochs,600000,
                                                          last_epoch = start_epoch - 1)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Scheduler state: {scheduler.state_dict()}")
        print(f"Last epoch in state: {scheduler.state_dict()['last_epoch']}")
        print(f"start epoch: {start_epoch}")
        print(f"Initial restored LR: {scheduler.get_last_lr()[0]}")
    else:
        scheduler = utils.get_cosine_schedule_with_warmup(optimizer, warm_up_epochs, 600000)
        start_epoch = 0

    print(f"total params:{sum(p.numel() for p in model.parameters())}")
    print(f"using lr {args.lr} and warm up epochs {warm_up_epochs}")
    model.criterion = criterion
    trains = train(model=model, criterion=criterion,start_epoch=start_epoch, num_epochs=args.num_epochs + start_epoch, optimizer=optimizer, scheduler=scheduler, batch_size=args.batch_size, seq_len=args.seq_len,
                   num_features=args.in_features,num_classes=args.num_outputs,cluster_type=args.cluster_type,
                   vary_feature=vary_feature, check_point=args.check_point, weight_concentration_prior=args.weight_concentration_prior,
                   mean_prior=args.mean_prior, mean_precision_prior=args.mean_precision_prior,
                   degrees_of_freedom_prior=args.degrees_of_freedom_prior,covariance_prior=args.covariance_prior,nan_frac = args.nan_frac)
    torch.save({
        'epoch': start_epoch + args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scheduler_last_epoch': scheduler.last_epoch,
    }, f"checkpoint_{args.check_point}_{args.mean_precision_prior}.pt")
    np.save(f"trains{start_epoch}.npy", trains)


