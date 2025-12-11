import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os

# --- 1. LIGHTWEIGHT BACKBONE (MobileNetV3 Block) ---
class Hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class MobileNetV3Block(nn.Module):
    '''
    Inverted Residual Block: The standard for efficient mobile inference.
    Expansion -> Depthwise Conv -> Pointwise Conv (Projection)
    '''
    def __init__(self, inp, oup, stride=1, expansion=4):
        super(MobileNetV3Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expansion != 1:
            # pw
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Hswish()
            ])
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Hswish(),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# --- 2. LIGHTWEIGHT UPSAMPLE (Removed PAM) ---
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.act = nn.PReLU()
        # REMOVED: self.pam (Too heavy for mobile)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        return x

class InfoGen(nn.Module):
    def __init__(self, t_emb, output_size):
        super(InfoGen, self).__init__()
        # Simplified InfoGen to reduce parameters
        self.tconv1 = nn.ConvTranspose2d(t_emb, 256, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.tconv2 = nn.ConvTranspose2d(256, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):
        # t_embedding shape: [Batch, Channels, 1, Width]
        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        return x

# --- 3. MAIN CLASS (Mobile Optimized + Dictionary Priors) ---
class RTSRN(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False, # Ignored in Mobile version
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 text_emb=37, 
                 out_text_channels=32,
                 triple_clues=False): # Added triple_clues arg to prevent crash if main calls it
        super(RTSRN, self).__init__()
        
        # A. Setup Inputs
        in_planes = 3
        if mask:
            in_planes = 4
        
        # B. Feature Extractor (Head)
        self.head = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # C. DICTIONARY PRIORS SETUP (Novelty)
        # We create a buffer for the matrix so it saves with the model state_dict
        # Shape 37x37 (Classes x Classes)
        self.register_buffer('priors', torch.zeros(text_emb, text_emb))
        self.priors_loaded = False
        self.num_classes = text_emb

        # D. BACKBONE: MobileNetV3 Blocks instead of GRU/ResBlock
        # Replaces MLPbasedRecurrentResidualBlockTL
        self.srb_nums = srb_nums
        mnet_blocks = []
        for i in range(srb_nums):
            # Input channels = hidden*2 + text_channels (fused) -> Output = hidden*2
            # Note: We do early fusion inside the block logic in forward, 
            # but here we define standard blocks.
            mnet_blocks.append(
                MobileNetV3Block(2 * hidden_units + out_text_channels, 2 * hidden_units + out_text_channels)
            )
        self.backbone = nn.ModuleList(mnet_blocks)
        
        # Projection layer to bring dimensions back if needed
        self.bottleneck = nn.Conv2d(2 * hidden_units + out_text_channels, 2 * hidden_units, 1)

        # E. Text Processing
        self.infoGen = InfoGen(text_emb, out_text_channels)

        # F. Upsampling (Tail)
        # Tentukan jumlah channel output: 4 jika pakai mask, 3 jika tidak.
        out_channels = 4 if mask else 3   # <--- LOGIKA DINAMIS
        
        upsample_block_num = int(math.log(scale_factor, 2))
        upsample_layers = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        
        # Gunakan variabel out_channels disini, JANGAN hardcode angka 3
        upsample_layers.append(nn.Conv2d(2 * hidden_units, out_channels, kernel_size=9, padding=4))
        
        self.tail = nn.Sequential(*upsample_layers)

        # --- AUTO-LOAD PRIORS (INTJ Strategy) ---
        # Automatically load priors when model is initialized
        self.load_priors('./quantized_priors.npy')

    def load_priors(self, path):
        """
        Loads the Quantized Bigram Matrix (.npy) created in Step 1.
        """
        if os.path.exists(path):
            print(f"Loading Dictionary Priors from {path}...")
            priors_np = np.load(path)
            
            # Normalize to 0-1 range if it was int8 (0-255)
            if priors_np.max() > 1.0:
                priors_np = priors_np.astype(np.float32) / 255.0
            
            # Assign to buffer
            self.priors.data = torch.from_numpy(priors_np).float().to(self.priors.device)
            self.priors_loaded = True
        else:
            print(f"Warning: Priors file not found at {path}. Using Identity matrix.")
            self.priors.data = torch.eye(self.num_classes).to(self.priors.device)

    def forward(self, x, text_emb=None, hint_ling=None, hint_vis=None):
        # Added hint_ling/hint_vis args to handle calls from legacy code if any
        
        # 1. Feature Extraction
        # x: [Batch, 4, H, W]
        feat = self.head(x)

        # 2. DICTIONARY PRIORS INJECTION (The Novelty)
        if text_emb is None:
            # Dummy embedding if not provided
            text_emb = torch.zeros(x.shape[0], self.num_classes, 1, 26).to(x.device)
        
        # text_emb shape is usually [Batch, 37, 1, SequenceLength] (Logits from CRNN)
        
        if self.priors_loaded:
            # Apply Bigram Correction: P(char_t) = P_crnn(char_t) * P_prior(char_t | char_t-1)
            # Simplified implementation: Matrix Multiplication over channel dimension
            
            # Reshape for matmul: [B, 37, Seq]
            B, C, H, W_text = text_emb.shape
            t_flat = text_emb.view(B, C, -1) 
            
            # Apply Priors: [37, 37] @ [B, 37, Seq] -> [B, 37, Seq]
            # This effectively mixes the character probabilities based on the bigram matrix
            rectified_emb = torch.matmul(self.priors, t_flat)
            text_emb = rectified_emb.view(B, C, H, W_text)

        # 3. Generate Spatial Text Features
        # InfoGen upsamples text embeddings to match image size roughly
        spatial_text = self.infoGen(text_emb)
        
        # Interpolate to match image feature size exactly
        spatial_text = F.interpolate(spatial_text, size=feat.shape[2:], mode='bilinear', align_corners=True)

        # 4. MobileNet Backbone with Early Fusion
        # Concat Image Features + Rectified Text Features
        # Shape: [B, Hidden*2 + TextChan, H, W]
        merged = torch.cat([feat, spatial_text], dim=1)
        
        # Pass through lightweight blocks
        for block in self.backbone:
            merged = block(merged)
            
        # Bottleneck to reduce channels back to Hidden*2
        res = self.bottleneck(merged)

        # Residual Connection (Global)
        res = res + feat

        # 5. Upsampling to High Resolution
        output = self.tail(res)
        
        # Tanh activation for image output (-1 to 1)
        output = torch.tanh(output)

        return output