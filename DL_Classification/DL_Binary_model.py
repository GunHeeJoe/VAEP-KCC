from torch.utils.data import Sampler

from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel

from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.dfm import DeepFactorizationMachineModel

from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.Upgrade_Dcn import Upgrade_Dcn

from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.afi import AutomaticFeatureInteractionModel

from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

import torch
import numpy as np
import pandas as pd
import random 

import os
import warnings
import tqdm
import copy


#TensorBoard : To compare loss and ROC AUC in real time
from torch.utils.tensorboard import SummaryWriter
CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime
now = datetime.now()
schedule = str(now.date()) + " " + str(now.hour) + "시 " + str(now.minute) + "분"

model_name = 'entitiy_dcn'
epoch = 20
learning_rate = 1e-4
weight_decay =  1e-2

name = f'runs/ now={schedule} & model={model_name} : epoch={epoch} & lr={learning_rate} & weight={weight_decay}'
writer = SummaryWriter(name)


class OverSampler(Sampler):
    def __init__(self, class_vector, batch_size):

        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector
        self.indices = list(range(len(self.class_vector)))
        self.batch_size = batch_size
        uni_label = torch.unique(class_vector)
        uni_label_sorted, _ = torch.sort(uni_label)

        # [0,1,2] 총 3개의 class_vector값이 labeling
        uni_label_sorted = uni_label_sorted.detach().numpy()
        label_bin = torch.bincount(class_vector.int()).detach().numpy()

        # 기존 train_data_loader에는 {0.0: 983346, 1.0: 13272, 2.0: 2159}개가 들어있으므로 dictionary형태로 변환
        label_to_count = dict(zip(uni_label_sorted, label_bin))

        # 전체개수=998,777=len(class_vector) // 전체 target개수
        # label_to_count[float(label)] : 각 label별 개수
        # 즉 각 target의 가중치를 계산함
        # target=0 : 998777 / 983347 = 1.015인 weight가 983346개 있음
        # target=1 : 998777 / 13272 = 75.25인 weight가 13272개 있음
        # target=2 : 998777 / 2159 = 462.61인 weight가 2159개 있음
        weights = [len(class_vector) / label_to_count[float(label)]
                       for label in class_vector]

        # 즉 target별 가중치를 이용해서 모든 target이 골고루 sampler되도록 sampling함
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.batch_size, replacement=True))

    def __len__(self):
        return len(self.class_vector)
    
def load_targets(data_loader):
    targets, predicts = list(), list()

    with torch.no_grad():
         for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, leave=True):
             targets.extend(target.tolist())

    return np.array(targets)

class CustomDataset():  
    # train, test dataset
    def get_xy(self, train_valid_test,score_concede_column):        
        # train_test_select==1 -> train dataset
        if train_valid_test == 1:
            print("\nmaking train_dataset")
            X = pd.read_csv('./soccer_binary_data/train/X_train',index_col=0)
            Y = pd.read_csv("./soccer_binary_data/train/Y_train",index_col=0) 
              
            indices = list(range(len(X)))
            random.shuffle(indices)

            # 섞인 인덱스를 사용하여 X_train과 Y_train을 매칭시킴
            X  = pd.DataFrame([X.loc[i] for i in tqdm.tqdm(indices,desc="X random shuffling")],columns=X.columns)
            Y  = pd.DataFrame([Y.loc[i] for i in tqdm.tqdm(indices,desc="Y random shuffling")],columns=Y.columns)
        elif train_valid_test == 2:
            print("\nmaking valid_dataset")
            X = pd.read_csv('././soccer_binary_data/valid/X_valid',index_col=0)
            Y = pd.read_csv("././soccer_binary_data/valid/Y_valid",index_col=0)           
        elif train_valid_test == 3:
            print("\nmaking test_dataset")
            X = pd.read_csv('././soccer_binary_data/test/X_test',index_col=0)
            Y = pd.read_csv("././soccer_binary_data/test/Y_test",index_col=0)           
        else:
            print("error")
                
        return X, Y[score_concede_column]
    
    def __init__(self, train_valid_test,score_concede_column):
        super(CustomDataset, self).__init__()

        X, Y = self.get_xy(train_valid_test,score_concede_column)

        #categorical, nunerical 데이터를 나눠서 저장한다음 합칠예정
        cate_X = pd.DataFrame()
        num_X = pd.DataFrame()
        
        #StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
        scaler = StandardScaler()

        #categorical(int) & continuos(float)으로 타입을 바꿔놓음 => 속성에 따른 표현을 달리 수행
        for col in tqdm.tqdm(X.columns,desc="Type converting"):
            #골득실은 int로 되어있지만, 실제로는 수치형속성이므로 numercal로 타입변환 시켜줌
            if col.startswith('goal'):
                X[col] = X[col].astype('float64')
                
            #bool, int32, int64타입은 categorical이므로 int32로 통일
            if X[col].dtype in ["bool", "int32",'int64']:
                cate_X[col] = X[col].astype("int32")
            #float64타입은 numerical데이터인데, 정규화 수행함
            elif X[col].dtype == 'float64':
                a = scaler.fit_transform(X[col].values.reshape(-1,1))
                num_X[col] = pd.DataFrame(a,columns=[col],index=X.index)
            else:
                print("error :", col, " : ", X[col].dtype)
         
        
        X = pd.concat([cate_X,num_X],axis=1)
  
        self.params = self.get_model_parameters(X)
        
        #가중치가 float32타입이므로 linear할 때, 타입오류뜸으로 float으로 통일함
        self.items = X.astype('float').to_numpy()
        self.targets = Y.astype('float').to_numpy()

        #categorical만 임베딩을 실시하는데, embed_input_size를 누적합으로도 사용할 때가 있음
        #본 연구에서는 사용하지 않았지만, 모든 임베딩 input size를 누적합으로 사용하는 방법도 있다.
        self.field_dims = np.max(cate_X, axis=0) + 1

        print("item.shape : ", self.items.shape)
        print("target.shape : ", self.targets.shape)

        self.user_field_idx = np.array((0, ), dtype=np.compat.long)
        self.item_field_idx = np.array((1,), dtype=np.compat.long)

    # Deep cross newtwork모델의 파라미터를 설정하는 함수
    def get_model_parameters(self, data):
        # Get categorical variable unique values
        categorical_cols = data.select_dtypes(
            include=['int32']).columns.tolist()
        
        #type클래스는 본 데이터에서(La Liga) 17,20번 action이 빠져있음
        #본 연구에서는 임베딩 input size를 각 컬럼 max개수 + 1로 설정함
        unique_values = []
        for col in categorical_cols:
            unique_values.append(data[col].max())
        
        # 앞에 어디까지가 categorical data인지 알기위한 변수
        # Count number of numerical variables
        numerical_cols = data.select_dtypes(
            include=['float64']).columns.tolist()
        
        num_numerical_features = len(numerical_cols)


        # set default embedding dimensions to 3
        # 모든 embed output size를 3으로 설정함
        embed_dim = self.get_embed_dim()
        
        # 모든 embed output size를 input-size의 절반으로 설정할 수 있지만, 여기서는 활용X
        # categorical_nuniques_to_dims = []
        # for unique in unique_values:
        #     target = int(unique*(2/4))
        #     if target < 2 :
        #         target = 2
        #     # if unique < 2 :
        #     #     unique = 2
        #     categorical_nuniques_to_dims.append([unique,target])
        
        categorical_nuniques_to_dims = list(
            zip(unique_values, [embed_dim] * len(unique_values)))
        
        #categorical_nuniques_to_dims = np.array(categorical_nuniques_to_dims)
        #fc_layers = len(categorical_nuniques_to_dims)*embed_dim + num_numerical_features
        
        #fc_layer도 사용할 수 있음(수치형 속성은 정규화만 수행하기로 함)
        fc_layers = num_numerical_features  
        fc_layers_construction = [fc_layers]
        
        dropout_probability = 0.2  # set default dropout probability to 0

        # Save all parameters in a dictionary
        model_params = {
            'categorical_nuniques_to_dims': categorical_nuniques_to_dims,
            'num_numerical_features': num_numerical_features,
            'fc_layers_construction': fc_layers_construction,
            'dropout_probability': dropout_probability,
            'categorical_feature' : categorical_cols,
            'numerical_cols' : numerical_cols
        }

        return model_params
    
    def get_embed_dim(self):
        embed_dim = 3
        return embed_dim
    
    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]

    def get_params(self):
        return self.params


def get_model(name, dataset):
    # Hyperparameters are empirically determined, not opitmized.
    field_dims = dataset.field_dims
    model_params = dataset.get_params()
    embed_dim = dataset.get_embed_dim()
    print("model name : ", name)
    
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=256)
    # elif name == 'hofm':
    #     return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    #elif name == 'dcn':
    #    return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'entitiy_dcn':
        return Upgrade_Dcn(model_params['categorical_nuniques_to_dims'], model_params['num_numerical_features'], model_params['fc_layers_construction'],
                           field_dims, embed_dim=embed_dim, num_layers=3,mlp_dims=(16,16), dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(128,64), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        # assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64, 64), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, epoch, device, log_interval=20):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, leave=True)

    model = model.to(device)
    
    for i, (fields, target) in enumerate(tk0):
        # print(pd.DataFrame(target.detach().cpu().numpy()).value_counts())
        fields, target = fields.to(device), target.to(device)

        # y prediction
        y = model(fields)

        loss = criterion(y, target.float())
        #target = target.long()
        #loss = criterion(y, target.float())
        #loss = criterion(y, target.squeeze(dim=-1))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            
            writer.add_scalar('Evaluation/train_loss', total_loss / log_interval, i + epoch * len(data_loader))
            total_loss = 0


def valid_evaluation(model, data_loader, criterion, epoch, device, log_interval=5):
    model.eval()
    targets, predicts = list(), list()
    
    total_loss = 0
    temp = []
    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, leave=True)):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            
            loss = criterion(y, target.float())
            
            total_loss += loss.item()
            temp.append(i + epoch * len(data_loader))
            
            if (i + 1) % log_interval == 0:                
                writer.add_scalar('valid_loss', total_loss, i + epoch * len(data_loader))                                
                total_loss = 0
            
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    # target shape: (batch, 1), predicts shape: (batch, 2) // score / not-score

    score_roc_auc = roc_auc_score(np.array(targets), np.array(predicts))
    brier_roc_auc = brier_score_loss(np.array(targets), np.array(predicts))
    
    print("score ROC AUC", score_roc_auc)
    print("score brier score", brier_roc_auc)
    for x in temp:
        #add_scaler(폴더이름,데이터, x좌표) : 1개의 값을 writer하는 것이고
        #add_scalers(폴더이름,dict, x좌표) : 여러개 값을 writer하는 것 => s붙이는것때매...개고생했네...
        writer.add_scalar('Valid ROC AUC', score_roc_auc, x)
        writer.add_scalar('Valid Brier score', brier_roc_auc, x)

    return score_roc_auc

def test_evaluation(model,data_loader, criterion, device):
    model.eval()
    targets, predicts = list(), list()
    
    total_loss = 0

    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, leave=True)):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            
            loss = criterion(y, target.float())
            total_loss += loss.item()
            
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    # target shape: (batch, 1), predicts shape: (batch, 2)
    score_roc_auc = roc_auc_score(np.array(targets), np.array(predicts))
    brier_roc_auc = brier_score_loss(np.array(targets), np.array(predicts))
    
    print("score ROC AUC", score_roc_auc)
    print("score brier score", brier_roc_auc)

    return predicts

    
def score_main(model_name, epoch, learning_rate, batch_size, weight_decay, device,save_dir):
    device = torch.device(device)
    #CustomDataset first parameter : train/valid/test
    #second parameter : score label / concede label
    train_dataset = CustomDataset(1,"scores") 
    valid_dataset = CustomDataset(2,"scores") 
    test_dataset = CustomDataset(3,"scores")
    
    target_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    target_vector = load_targets(target_data_loader)

    # sampler는 oversample이라기 보다는 데이터를 batch단위로 어떠한 규칙으로 추출할지 정하는 과정임
    # oversampler클래스에서는 각 target이 밸런스있기 뽑힐 수 있도록 weight를 조정해서
    # 모든 target이 골고루 뽑힐 수 있게 조정함
    # ex) batch=4096단위로 학습할 때, ex{ 0: 1369, 1: 1312, 2: 1415}씩 뽑아서 train데이터에 활용함
    sampler = OverSampler(class_vector=torch.from_numpy(
        target_vector).view(-1), batch_size=batch_size * 244)
    
    
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8,sampler=sampler)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=8,shuffle=False)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8,shuffle=False)
    

    print("\nscore Deep learning start \n")
    
    #다양한 deep learning모델를 활용할 수 있고, 본 연구에서는 Deep cross network모델를 활용함
    model = get_model(model_name, train_dataset).to(device)
    
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # early_stopper = EarlyStopper(num_trials=20)

    early_stopper = EarlyStopper(num_trials=20, save_path=f"{save_dir}/{model_name}.pt")

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, epoch_i, device)
        auc = valid_evaluation(model, valid_data_loader, criterion, epoch_i, device)

        print('epoch : ', epoch_i)
        if not early_stopper.is_continuable(model, auc):
            print(f'early validation: best auc: {early_stopper.best_accuracy}')
            break

    writer.close()
    #test_data_loader에서는 loss검사를 똑같이 할 필요없음
    score_probability = test_evaluation(model,test_data_loader, criterion, device)
    print("score Deep learning end \n")
    
    return pd.DataFrame(score_probability,columns=['scores'])

def concede_main(model_name, epoch, learning_rate, batch_size, weight_decay, device,save_dir):
    device = torch.device(device)
    #CustomDataset first parameter : train/valid/test
    #second parameter : score label / concede label
    train_dataset = CustomDataset(1,"concedes") 
    valid_dataset = CustomDataset(2,"concedes") 
    test_dataset = CustomDataset(3,"concedes")
    
    target_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    target_vector = load_targets(target_data_loader)

    # sampler는 oversample이라기 보다는 데이터를 batch단위로 어떠한 규칙으로 추출할지 정하는 과정임
    # oversampler클래스에서는 각 target이 밸런스있기 뽑힐 수 있도록 weight를 조정해서
    # 모든 target이 골고루 뽑힐 수 있게 조정함
    # ex) batch=4096단위로 학습할 때, ex{ 0: 1369, 1: 1312, 2: 1415}씩 뽑아서 train데이터에 활용함
    sampler = OverSampler(class_vector=torch.from_numpy(
        target_vector).view(-1), batch_size=batch_size * 244)
    
    
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8,sampler=sampler)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=8,shuffle=False)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8,shuffle=False)
    

    print("concede Deep learning start \n")
    
    #다양한 deep learning모델를 활용할 수 있고, 본 연구에서는 Deep cross network모델를 활용함
    model = get_model(model_name, train_dataset).to(device)
    
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # early_stopper = EarlyStopper(num_trials=20)

    early_stopper = EarlyStopper(num_trials=20, save_path=f"{save_dir}/{model_name}.pt")

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, epoch_i, device)
        auc = valid_evaluation(model, valid_data_loader, criterion, epoch_i, device)

        print('epoch : ', epoch_i)
        if not early_stopper.is_continuable(model, auc):
            print(f'early validation: best auc: {early_stopper.best_accuracy}')
            break

    writer.close()
    #test_data_loader에서는 loss검사를 똑같이 할 필요없음
    concede_probability = test_evaluation(model,test_data_loader, criterion, device)
    print("concede Deep learning end \n")
    
    return pd.DataFrame(concede_probability,columns=['concedes'])
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default=model_name)
    parser.add_argument('--epoch', type=int, default=epoch)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    print(args.device)
    
    score_probability = score_main( 
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir
         )
    
    concede_probability = concede_main( 
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir
         )

    df = pd.concat([score_probability,concede_probability],axis=1)
    print(df)
    df.to_csv("./result/Dcn_binary_result")