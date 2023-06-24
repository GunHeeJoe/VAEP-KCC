import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        
        #batch=4096개의 데이터가 있고, 각 feature개수 151개이므로
        #embedding size는 torch.Size([4096,151,64])형태로 임베딩 됨
    
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        #self.linear에서만 output_dim=3의 값이 추출되고
        #거기에 self.fm(self.embedding(x))의 각 데이터별 1개의 값이 추출됨
        self.linear = FeaturesLinear(field_dims)
        #self.linear = FeaturesLinear(field_dims,categorical_nuniques_to_dims,num_numerical_features,fc_layers_construction,self.dropout_probability)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # print("self.embedding(x) size: ",self.embedding(x).shape)
        # print("self.fm(self.embedding(x)) size: ",self.fm(self.embedding(x)).shape)
        # print("self.embedding(x) : ",self.embedding(x)[0])
        # print("self.fm(self.embedding(x)) : ",self.fm(self.embedding(x))[0])
        
        #self.embedding(x) // #batch=4096개의 데이터가 있고, 각 feature개수 151개이므로
        #embedding size는 torch.Size([4096,151,64])형태로 임베딩 됨
        
        #self.fm(self.embedding(x)) // [4096,151,64]형태 즉 한개의 데이터는 64개로 나열된 embedding값이 들어있음
        #해당64개값 -> 1개의 값으로 FactorizationMachine을 적용시킴 => [4096,1]형태로 바뀜
        #self.fm(self.embedding(x))은 각 데이터별로 1개의 값이 추출됨
        # print("self.embedding(x) : " , self.embedding(x).shape)
        # print("self.linear(x) : ",self.linear(x).shape)
        # print("self.fm(self.embedding(x)) : ",self.fm(self.embedding(x)).shape)
        # print("(self.linear(x) + self.fm(self.embedding(x))) : ",(self.linear(x) + self.fm(self.embedding(x))).shape)
        # print("self.embedding(x) : " , self.embedding(x)[0])
        # print("self.linear(x) : ",self.linear(x))
        # print("self.fm(self.embedding(x)) : ",self.fm(self.embedding(x)))
        # print("(self.linear(x) + self.fm(self.embedding(x))) : ",(self.linear(x) + self.fm(self.embedding(x))))
        
        x = self.linear(x) + self.fm(self.embedding(x))
      
        #return torch.sigmoid(x.squeeze(1))

        #각 class별 Probability prediction값을return
        print(torch.nn.functional.softmax(x, dim=-1).shape)
        print(torch.nn.functional.softmax(x, dim=-1))
        ss
        return torch.nn.functional.softmax(x, dim=-1)
