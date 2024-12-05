import torch 
import torch.nn as nn 
import math
#512 is th d_model that i am using


class InputEmbedding(nn.Module):
    def __init__(self,d_model: int,vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model: int,seq_len: int,dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #matrix of shape of (seq,d_model)
        pe = torch.zeros(seq_len,d_model)
        #create a vector of shape (seq)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #(seq,1)-->pos
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0/d_model))) # 10000^(21/dmodel) formula
        #apply sin for even and cosine for odd
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        
        pe.unsqueeze(0) #(1,seq,d_model)
        #we will register this tensor in the buffer of the logic -->to save it along with the model file but not as parameters
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False)# this tensor is not learned
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    def __init__(self,eps: float=10**-6) -> None: #epsilon for numerical stbility limiting too much float size
        super().__init__()
        self.eps = eps
        self.alpha =  nn.Parameter(torch.ones(1)) #gamma that is multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #beta para that is added
        
    def forward(self,x):
        mean = x.mean(dim = -1,keepdim=True)#batch norm
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x.mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float) ->None:
        super().__init__()    
        self.linear_1 = nn.Linear(d_model,d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) #w2 and b2
        
    def forward(self,x):
        #input (Batch,seq,d_model) --> (Batch,seq,d_ff) -> (Batch,seq,D_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model: int,h: int,dropout: float)->None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model is not divisble by h" # to check if d_model is actually divisibe by h to check decay
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod #call this method to be called with class.method way
    def Attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_scores =  (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask==0,-1e9) #we use this to remove the filler words 
        attention_scores = attention_scores.softmax(dim = -1) #(Batch , h ,seq,seq)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value) , attention_scores #second one help in visualization
    
        
    def forward(self,q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        #(Batch,seq,d_model)->(Batch,seq,h,d_k)-->(Batch,h,seq,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)# we wanna only change teh last layer d_model that contains the embed
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)  
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)  
        
        x,self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        # (Batch , h ,seq , d_k) --> (Batch,seq,h,d_k) --> (Batch , seq , d_model)
        #we will use contiguous before the view to put it in memory similar to inplace
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        return self.w_o(x)

#now we need a layer to take a skip connection layer-->Residual Connection 

class ResCon(nn.Module):
    def __init__(self,dropout: float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()
        
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
#now we need to construct N X Encoder blocks

class EncoderBlock(nn.Module):
    def __init__(self,self_attention : MultiHeadAttention , feed_forward:FeedForward,dropout:float)->None:
        super().__init__()
        self.dropout = dropout
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.res_con = nn.ModuleList(
            [ResCon(dropout) for _ in range(2)]
        )
        
    def forward(self,x,src_mask):#we don't want padding words to interact with the other words
        x = self.res_con[0](x,lambda x:self.self_attention(x,x,x,src_mask))
        x = self.res_con[1](x,self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) ->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()
        
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
    
###now let's build the decoder part

# Output(shifted right)--> class is similar to that of input embeddings
#same class for Positional Encoding

## DECODER = DecoderBlock*N

class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention , cross_attention: MultiHeadAttention , feed_forward:FeedForward,dropout:float)-> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.cross_attention = cross_attention
        self.res_con = nn.ModuleList([ResCon(dropout) for _ in range(3)])
        
    def forward(self,x,encoder_output,src_mask,tgt_mask): #src_mask and tgt_mask since tow languages are there one from encoder another in decoder itself
        x = self.res_con[0](x,lambda x:self.self_attention(x,x,x,tgt_mask))
        x = self.res_con[1](x,lambda x:self.cross_attention(x,encoder_output,encoder_output,src_mask))
        #src mask is the mask from the encoder
        x = self.res_con[2](x,lambda x:self.feed_forward)
        return x
    
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__()
        self.norm = LayerNorm()
        self.layers = layers
         
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
    

#linear layer or projection layer that projects the embedding to the vocabulary
class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int)->None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        #(Batch , seq , d_model) --> (Batch,seq,Vocab_size)
        #for numerical stability we will add log softmax
        return torch.log_softmax(self.proj(x),dim=-1)
    

## TRANSFORMER BLOCK

class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embedding: InputEmbedding , tgt_embedding:InputEmbedding , src_pos:PositionalEmbedding , tgt_pos:PositionalEmbedding , projection_layer:ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embedding
        self.tgt_embed = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encoder(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
        
    def decoder(self,encoder_output,tgt,src_mask,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    
    
#last block to be built . we are done building all the blocks  this is a method that combines all and gives hyperparameter for all and builds the transformer 
#the application of transformer changes here with this method -->here language translation
#N->number of encoder decoders
def Build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int = 512, N: int = 6,h:int=8, dropout=0.1,d_ff:int=2048)->Transformer:#the length varies since the languages differ
    src_embed = InputEmbedding(d_model,src_vocab_size)
    tgt_embed = InputEmbedding(d_model,tgt_vocab_size)
    
    src_pos = PositionalEmbedding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEmbedding(d_model,tgt_seq_len,dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForward(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
        
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForward(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention,decoder_cross_attention,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
    
    #can now create teh encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)
    #let's create the transformer
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    
    #initialize the parameters -- weight initialization techniques use xavier or He
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
            
    return transformer


    
    
    
        
                
                    
        
           
        
        
            
                
    
    
         