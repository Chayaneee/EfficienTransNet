### ----- Base packages ----- ###
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

### ----- PyTorch packages ----- ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

### ----- Transformer Packages ----- ###
import timm
import transformers
from transformers import AutoModel

### ----- Helper Packages ----- ###
from tqdm import tqdm
import wandb

### ----- Project Packages ----- ###
from utils import save, load, train, test, data_to_device, data_concatenate
from datasets import MIMIC, IUCXR
from losses import CELoss, CELossTotal, CELossTotalEval, CELossTransfer, CELossShift
from models_deit import ViT, MViT, TNN, Classifier, Generator, ClsGen
from baseline.transformers.models import Transformer

### ----- Helper Functions ----- ###
def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    return threshold[ix]

def infer(data_loader, model, device='cpu', threshold=None):
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)

            # Use single input if there is no indication/clinical history
            if threshold != None:
                output = model(image=source[0], history=source[3], threshold=threshold)
                
            else:
                output = model(source[0])
                
            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)
    
    return outputs, targets

### ----- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=123)


RELOAD = False # True / False
PHASE = 'TRAIN' # TRAIN / TEST / INFER
DATASET_NAME = 'MIMIC' # IUCXR / MIMIC 
BACKBONE_NAME = 'DeiT-384' # DeiT-384/EfficientNetB7
MODEL_NAME = 'ClsGen'

if DATASET_NAME == 'MIMIC':
    EPOCHS = 50 # Start overfitting after 20 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64 # 128 # Fit 4 GPUs
    MILESTONES = [25] # Reduce LR by 10 after reaching milestone epochs
    
elif DATASET_NAME == 'IUCXR':
    EPOCHS = 50 # Start overfitting after 20 epochs
    BATCH_SIZE = 4 if PHASE == 'TRAIN' else 64 # Fit 4 GPUs
    MILESTONES = [25] # Reduce LR by 10 after reaching milestone epochs
    
else:
    raise ValueError('Invalid DATASET_NAME')


#Wandb Initialization

wandb.init(
      project= "TransXplain-Net-Project",
      config = {
          "Learning_rate":"LR",
          "Dataset": "DATASET_NAME",
          "model":"MODEL_NAME",
          "epochs":"EPOCHS",
         }
     )


if __name__ == "__main__":
    # --- Choose Inputs/Outputs
    if MODEL_NAME in ['ClsGen']:
        SOURCES = ['image','caption','label','history']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','label','history']
        KW_TGT = None
        KW_OUT = None
                
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')
        
    # --- Choose a Dataset ---
    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (384,384)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2
        
        dataset = MIMIC('/mnt/data/chayan/MIMIC_CXR_JPG/physionet.org/files/mimic-cxr-jpg/2.0.0/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=True, debug_mode=False, train_phase=(PHASE == 'TRAIN'))
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
            
    elif DATASET_NAME == 'IUCXR':
        INPUT_SIZE = (384,384)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2

        dataset = IUCXR('/home/chayan/2.CGI/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
        
    else:
        raise ValueError('Invalid DATASET_NAME')

    # --- Choose a Backbone ---
     
    if BACKBONE_NAME == 'DeiT-384':
        backbone = timm.create_model('deit_base_patch16_384', pretrained=True)
        #backbone =AutoModel.from_pretrained("facebook/deit-base-patch16-384")
        #       
        checkpoint = torch.load("/mnt/data/chayan/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/results/latest_Model.pth")
## load model weights state_dict
            
        n_inputs = backbone.head.in_features
        backbone.head = nn.Sequential(
           nn.Linear(n_inputs, 512),
           nn.ReLU(),
           nn.Dropout(0.5),
           nn.Linear(512, 14)
        )
### If state_dict_key error occur
        state_dict = checkpoint['model_state_dict']
#        print(state_dict)
        
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

#        new_state_dict = {}
#        for key, value in state_dict.items():
#            if key == "head.0.weight":
#                new_state_dict["head.weight"] = value
#            elif key == "head.0.bias":
#                new_state_dict["head.bias"] = value
#            else:
#                new_state_dict[key] = value

        backbone.load_state_dict(state_dict)  ### new_state_dict if key error
        
#        for param in backbone.parameters():
#            param.requires_grad = False
       
        for name, param in backbone.named_parameters():
          if 'attn' in name:
             param.requires_grad = True
          elif 'head' in name:
             param.requires_grad = True
          else:
             param.requires_grad = False
  
        for name, param in backbone.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")        
        
        FC_FEATURES = 768
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')

    # --- Choose a Model ---
    if MODEL_NAME == 'ClsGen':
        LR = 3e-4 # Fastest LR
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        FWD_DIM = 256
        
        NUM_HEADS = 8
        NUM_LAYERS = 1
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        #vit = ViT(backbone, BACKBONE_NAME)
        #vit = MViT(vit)
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        
        # Not enough memory to run 8 heads and 12 layers, instead 1 head is enough
        NUM_HEADS = 1
        NUM_LAYERS = 12
        
        cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, vit=None, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
        gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)
        
        model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
        criterion = CELossTotal(ignore_index=3)
        
    else:
        raise ValueError('Invalid MODEL_NAME')
    
    # --- Main program ---
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    print('Total Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad ))
    
    last_epoch = -1
    best_metric = 1e9

    checkpoint_path_from = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    checkpoint_path_to = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    
    if RELOAD:
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler) # Reload
        # last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model) # Fine-tune
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))

    if PHASE == 'TRAIN':
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(last_epoch+1, EPOCHS):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, scaler=scaler)
            val_loss = test(val_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            test_loss = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            
            wandb.log({"Train_Loss":train_loss, "Val_Loss": val_loss})

            scheduler.step()
           # scheduler.step(val_loss)
            
            if best_metric > val_loss:
                best_metric = val_loss
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print('New Best Metric: {}'.format(best_metric)) 
                print('Saved To:', checkpoint_path_to)
    
    elif PHASE == 'TEST':
        # Output the file list for inspection
        out_file_img = open('outputs/{}_{}_{}_{}_Img.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        for i in range(len(test_data.idx_pidsid)):
            out_file_img.write(test_data.idx_pidsid[i][0] + ' ' + test_data.idx_pidsid[i][1] + '\n')
            
       
    elif PHASE == 'INFER':
        txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.25)
        gen_outputs = txt_test_outputs[0]
        gen_targets = txt_test_targets[0]
        
        out_file_ref = open('outputs/x_{}_{}_{}_{}_Ref.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_hyp = open('outputs/x_{}_{}_{}_{}_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_lbl = open('outputs/x_{}_{}_{}_{}_Lbl.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        
        for i in range(len(gen_outputs)):
            candidate = ''
            for j in range(len(gen_outputs[i])):
                tok = dataset.vocab.id_to_piece(int(gen_outputs[i,j]))
                if tok == '</s>':
                    break # Manually stop generating token after </s> is reached
                elif tok == '<s>':
                    continue
                elif tok == '▁': # space
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' '
                elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' ' + tok + ' ' 
                    else:
                        candidate += tok + ' '
                else: # letter
                    candidate += tok       
            out_file_hyp.write(candidate + '\n')
            
            reference = ''
            for j in range(len(gen_targets[i])):
                tok = dataset.vocab.id_to_piece(int(gen_targets[i,j]))
                if tok == '</s>':
                    break
                elif tok == '<s>':
                    continue
                elif tok == '▁': # space
                    if len(reference) and reference[-1] != ' ':
                        reference += ' '
                elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
                    if len(reference) and reference[-1] != ' ':
                        reference += ' ' + tok + ' ' 
                    else:
                        reference += tok + ' '
                else: # letter
                    reference += tok    
            out_file_ref.write(reference + '\n')

        for i in tqdm(range(len(test_data))):
            target = test_data[i][1] # caption, label
            out_file_lbl.write(' '.join(map(str,target[1])) + '\n')
                
    else:
        raise ValueError('Invalid PHASE')
