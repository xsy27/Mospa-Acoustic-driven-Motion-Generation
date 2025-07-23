import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch import Tensor


class MotionDiffusion(nn.Module):
    def __init__(self, pose_vec, vec_len, audio_dim, clip_len=240, binaural=False,
                 latent_dim=512, ff_size=1024, num_layers=4, num_heads=8, dropout=0.2,
                 activation="gelu", legacy=False, 
                 arch='trans_enc', cond_mask_prob=0, mask_sound_source=False, mask_genre=False, device='cpu'):
        super().__init__()

        self.legacy = legacy
        self.training = True
        
        self.binaural = binaural
        self.pose_vec = pose_vec
        self.vec_len = vec_len
        self.audio_dim = audio_dim
        self.clip_len = clip_len

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.cond_mask_prob = cond_mask_prob
        self.mask_sound_source = mask_sound_source
        self.mask_genre = mask_genre
        self.arch = arch
        self.device = device

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_audio = AudioEmbedder(self.audio_dim, self.latent_dim)
        self.embed_motion = MotionEmbedder(self.vec_len, self.latent_dim)
        self.embed_ss = SoundSourceEmbedder(self.latent_dim)
        self.embed_state = StateEmbedder(self.latent_dim)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqEncoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == 'gru':
            print("GRU init")
            self.seqEncoder = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
      
        self.output_process = OutputProcess(self.vec_len, self.latent_dim)

    def forward(self, x, timesteps, y=None):
        bs, vec_len, nframes = x.shape
        if y != None:
            audio_feature = y['audio'] # the audio features; shape: [bs, aud_dim, nframes]
            pred_ss = y['pred_ss'] # the predicted approcimate sound source direction; shape: [bs, 3, nframes]
            genre = y['state'] # the state of the charatcer: 0, 1, 2; shape: [bs, 1]
        else:
            raise ValueError("No Condition!")
        
        # mask sound source
        if self.mask_sound_source:
            pred_ss = torch.zeros_like(pred_ss)
        else:
            pred_ss = F.normalize(pred_ss, dim=-1)
        # mask state
        if self.mask_genre:
            genre = torch.zeros_like(genre)

        # mask
        keep_batch_idx = torch.rand(bs, device=audio_feature.device) < (1-self.cond_mask_prob)
        audio_feature = audio_feature * keep_batch_idx.view((bs, 1, 1))

        time_emb = self.embed_timestep(timesteps)  # [1, bs, L]
        audio_emb = self.embed_audio(audio_feature) # [nframes, bs, L]
        motion_emb = self.embed_motion(x) # [nframes, bs, L]
        ss_emb = self.embed_ss(pred_ss) # [nframes, bs, L]
        genre_emb = self.embed_state(genre).unsqueeze(0) # [1, bs, L]
        
        xseq = torch.cat((audio_emb, ss_emb, genre_emb, time_emb, motion_emb), axis=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:]
        output = self.output_process(output)
        return output

# Embed timestep
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

# Embed audio features
class AudioEmbedder(nn.Module):
    def __init__(self, audio_dim, latent_dim):
        super().__init__()
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.audioEmbedding = nn.Sequential(
            nn.Linear(self.audio_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_dim)
        )
    
    def forward(self, x):
        bs, audio_dim, nframes = x.shape
        x = x.permute((2, 0, 1))
        x = self.audioEmbedding(x)
        return x

# Embed motion sequences
class MotionEmbedder(nn.Module):
    def __init__(self, vec_len, latent_dim):
        super().__init__()
        self.vec_len = vec_len
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.vec_len, self.latent_dim)

    def forward(self, x):
        bs, vec_len, nframes = x.shape
        x = x.permute((2, 0, 1))
        x = self.poseEmbedding(x)  
        return x

# Embed predicted sound source direction of arrival
class SoundSourceEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.ssEmbedding = nn.Linear(3, self.latent_dim)

    def forward(self, x):
        bs, _, nframes = x.shape
        x = x.permute((2, 0, 1))
        x = self.ssEmbedding(x)  
        return x

# Embed state of the character
class StateEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.state_embed = nn.Parameter(torch.randn(3, latent_dim))

    def forward(self, state):
        idx = state.to(torch.long) 
        output = self.state_embed[idx]
        return output

# PE
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:x.shape[0]] if not self.batch_first else self.pe[:x.shape[1]].permute(1, 0, 2))
        return self.dropout(x)

# Convert the output shape to [batch_size, pose_vector_length, frame_number]
class OutputProcess(nn.Module):
    def __init__(self, vec_len, latent_dim):
        super().__init__()
        self.vec_len = vec_len
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.vec_len)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)
        output = output.reshape(nframes, bs, self.vec_len)
        output = output.permute(1, 2, 0)
        return output

class Extractor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, input_type=""):
        super(Extractor, self).__init__()
        
        # Bi-GRU
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_type = input_type

        if self.input_type == "audio":
            latent_dim = 1024
            self.audio_embedder = nn.Linear(input_dim-4, latent_dim-4)
            self.encoder = nn.GRU(latent_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        elif self.input_type == "motion":
            latent_dim = 512

            self.embedding = nn.Linear(input_dim, latent_dim)
            self.pos_encoder = PositionalEncoding(latent_dim, batch_first=True)
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=4,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            )
            self.motion_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
            
            # Transformer decoder
            decoder_layers = nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            )
            self.motion_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
            self.fc_out = nn.Linear(latent_dim, input_dim)

            self.encoder = nn.GRU(latent_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
    
    def forward(self, input): 
        if self.input_type == "audio":
            encoded_audio = self.audio_embedder(input[..., :-4])
            output, hidden = self.encoder(torch.cat([encoded_audio, input[..., -4:]], dim=-1))  # hidden: (num_layers*2, B, hidden_size)
            decoded = None
        elif self.input_type == "motion":
            B, T, D = input.shape

            # encode motion sequences to (B, T, 512)
            embedded = self.embedding(input)  # (B, T, latent_dim)
            embedded = self.pos_encoder(embedded)
            z = self.motion_encoder(embedded)  # (B, T, latent_dim)

            # use the encoded motion sequences to get the motion features
            output, hidden = self.encoder(z)
            
            # decode the encoded motion sequence to perform reconstruction loss
            latent = z.mean(dim=1)  # (B, latent_dim)
            latent_expanded = latent.unsqueeze(1).repeat(1, T, 1)  # (B, T, latent_dim)
            tgt = self.pos_encoder(latent_expanded)
            decoded = self.motion_decoder(
                tgt=tgt,
                memory=z
            )  # (B, T, latent_dim)
            decoded = self.fc_out(decoded)  # (B, T, D)
        
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        last_forward = hidden[-1, 0]  # (B, hidden_size)
        last_backward = hidden[-1, 1]  # (B, hidden_size)
        features = torch.cat([last_forward, last_backward], dim=1)  # (B, 2048)

        return decoded, features

class AudioExtractor(Extractor):
    def __init__(self, audio_dim, hidden_size, num_layers):
        super().__init__(audio_dim, hidden_size, num_layers, input_type="audio")

class MotionExtractor(Extractor):
    def __init__(self, motion_dim, hidden_size, num_layers):
        super().__init__(motion_dim, hidden_size, num_layers, input_type="motion")