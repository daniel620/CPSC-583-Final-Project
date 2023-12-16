from calendar import c
from torch_geometric.datasets import ZINC, PPI
from torch_geometric.loader import DataLoader

import torch_geometric.transforms as T
from SPD_encoder import AddShortestPathMatrix, SPDEncoder, RemoveFeatures
import os.path as osp

from torch_geometric.nn import GINEConv, GPSConv

import torch
import warnings

# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.data.in_memory_dataset")



def main():
    # Set device to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)    

    torch.cuda.empty_cache()

    print(f"Using device: {device}")
    
    # data transform
    transform = RemoveFeatures()
    pre_transform = T.Compose([AddShortestPathMatrix(), T.AddRandomWalkPE(20, attr_name='rwpe')])
    

    # ZINC
    ZINC_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ZINC-SPD')  
    print("Data path:", ZINC_path)
    # Load datasets
    train_dataset = ZINC(ZINC_path, subset=True, split='train', pre_transform=pre_transform, transform=transform)
    val_dataset = ZINC(ZINC_path, subset=True, split='val', pre_transform=pre_transform, transform=transform)
    # test_dataset = ZINC(ZINC_path, subset=True, split='test', pre_transform=transform, transform=transform)
    
    # PPI
    PPI_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PPI-SPD')
    PPI_train_dataset = PPI(PPI_path,  split='train', pre_transform=pre_transform, transform=transform)
    PPI_val_dataset = PPI(PPI_path, split='val', pre_transform=pre_transform, transform=transform)
    # PPI_test_dataset = PPI(PPI_path, split='test', pre_transform=transform, transform=transform)


    # Concatenate datasets
    # train_dataset = train_dataset + PPI_train_dataset
    # val_dataset = val_dataset + PPI_val_dataset
    train_dataset = PPI_train_dataset
    val_dataset = PPI_val_dataset

    # Create data loader
    BATCH_SIZE = 1000
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # print loader info
    print("Train dataset:", len(train_dataset), 'Average graph size:', train_dataset.data.num_nodes / len(train_loader), 'Average edge size:', train_dataset.data.num_edges / len(train_loader))
    print("Val dataset:", len(val_dataset), 'Average graph size:', val_dataset.data.num_nodes / len(val_loader), 'Average edge size:', val_dataset.data.num_edges / len(val_loader))

    # Initialize model and optimizer
    model = SPDEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Training loop
    model.train()
    for epoch in range(1, 50):
        # print(epoch)
        total_loss, total_count_pairs = 0.0, 0
        for batch in train_loader:
            # batch = batch.to(device)  # Move batch to GPU
            # 除了 batch.spd，其他.to(device)

            for key in batch.keys():
                if key != 'spd' and key != 'num_nodes':
                    batch[key] = batch[key].to(device)

            model.GPS.redraw_projection.redraw_projections()  # Ensure this also works with CUDA
            optimizer.zero_grad()

            out = model(batch.rwpe, batch.edge_index, batch.batch)
            loss, count_pairs = model.batch_loss(out, batch.spd, batch.ptr, batch.batch_size)
            # print(loss)
            total_loss += loss.item()
            loss = loss / count_pairs
            loss.backward()
            optimizer.step()
            total_count_pairs += count_pairs
            # total_loss += loss.item()

                # Print training loss
        # train_loss = total_loss / len(train_loader)
        train_loss = total_loss/total_count_pairs 
        print(f"Epoch: {epoch}, Training Loss: {train_loss:.6f}", end='')

        # Evaluation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            test_loss, total_count_pairs = 0.0, 0
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.rwpe, batch.edge_index, batch.batch)
                loss, count_pairs = model.batch_loss(out, batch.spd, batch.ptr, batch.batch_size, max_pairs_per_graph=-1)
                test_loss += loss.item()
                total_count_pairs += count_pairs
                # test_loss += loss.item()
            # test_loss /= len(val_loader) 
            test_loss = test_loss / total_count_pairs 
        # Print test loss
        print(f", Test Loss: {test_loss:.6f}  {count_pairs}")
        model.train()  # Set model back to train mode

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    torch.save(model.state_dict(), f'model_weights_{timestamp}.pth')

if __name__ == "__main__":
    main()