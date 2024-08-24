import torch
from torch import nn

# class Permute(nn.Module):
#     def __init__(self, *dims):
#         super(Permute, self).__init__()
#         self.dims = dims

#     def forward(self, x: torch.Tensor):
#         return x.permute(self.dims)

class Model(nn.Module):
    """
    ex_ms2
    spectrum
    mask
    featured_ion
    """
    def __init__(self, num_layers: int, d_in: int, feedward_dim: int, n_head: int, d_out: int, dropout: float=0.1) -> None:
        super().__init__()
        self.coeff_attention = nn.MultiheadAttention(feedward_dim, n_head, batch_first=True, dropout=dropout)
        self.blend_attention = nn.MultiheadAttention(feedward_dim, n_head, batch_first=True, dropout=dropout)

        self.selu = nn.SELU()

        self.coeff_block = nn.Sequential(
            # nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in),
            nn.ReLU()
        )

        self.pep_dimtrans= nn.Sequential(
            nn.Linear(d_in, feedward_dim),
            nn.SELU()
        )

        self.ex_dimtrans = nn.Sequential(
            nn.Linear(d_in, feedward_dim),
            nn.SELU()
        )

        self.featured_ion_dimtrans = nn.Sequential(
            nn.Linear(d_in, feedward_dim),
            nn.SELU()
        )

        self.ppm_dimtrans = nn.Sequential(
            nn.BatchNorm1d(num_features=d_in),
            nn.Linear(d_in, feedward_dim),
            nn.SELU()
        )

        self.xic_corr_dimtrans = nn.Sequential(
            nn.Linear(d_in, feedward_dim),
            nn.SELU()
        )

        self.ex_encoder = nn.Sequential(
            # nn.LayerNorm(feedward_dim),
            nn.LSTM(
                num_layers=num_layers,
                input_size=feedward_dim,
                hidden_size=feedward_dim,
                batch_first=True,
                dropout=dropout,
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(feedward_dim, feedward_dim // 2),
            nn.SELU(),
            nn.Linear(feedward_dim // 2, d_out),
            nn.Sigmoid()
        )

    def forward(self, ex_ms2, spectrum, mask, xic_corr, ppm, featured_ion):
        """
        ex_ms2: batch * seq_len * d_in * 2
        spectrum: batch * 1 * d_in * 2
        mask: batch * seq_len
        xic_corr: batch * d_in
        ppm: batch * seq_len * d_in
        featured_ion: batch * d_in
        """
        featured_ion = torch.unsqueeze(featured_ion, dim=1)
        xic_corr = torch.unsqueeze(xic_corr, dim=1)

        featured_ion = self.featured_ion_dimtrans(featured_ion)
        xic_corr = self.xic_corr_dimtrans(xic_corr)
        ppm = self.ppm_dimtrans(ppm)

        ex_ms2, spectrum = ex_ms2[:, :, :, 1], spectrum[:, :, :, 1]

        ex_ms2 = self.coeff_block(ex_ms2)
        ex_ms2 = self.ex_dimtrans(ex_ms2)
        ex_ms2, _ = self.ex_encoder(ex_ms2)
        ex_ms2 = self.selu(ex_ms2)
        ex_ms2, ex_ms2_attention_weight = self.coeff_attention(ex_ms2, ex_ms2, ex_ms2, key_padding_mask=mask)

        spectrum = self.pep_dimtrans(spectrum)
        spectrum = torch.concat((spectrum, xic_corr, ppm, featured_ion), dim=1)
        x, spectrum_ex_ms2_attention_weight = self.blend_attention(spectrum, ex_ms2, ex_ms2, key_padding_mask=mask)
        
        x = torch.mean(x, axis=1)

        # x = torch.max(x, dim=1)[0]

        x = self.decoder(x)

        return x