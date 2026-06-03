import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_rnn_core import BaseConvSequenceModel


class GaWFRNNConv(BaseConvSequenceModel):
    """
    GaWF (Gated with Feedback) RNN Model.
    Encoder and classifier from BaseConvSequenceModel. Forward overridden for feedback.
    """

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        feedback_dim=None,
    ):
        super(GaWFRNNConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            hidden_size=hidden_size,
            max_chars=15,
            predict_all_chars=False,
        )
        self.num_classes = num_classes
        self.num_pos = num_pos
        self.hidden_size = hidden_size
        self.output_feedback_dim = num_classes + num_pos
        self.feedback_dim = (
            self.output_feedback_dim if feedback_dim is None else int(feedback_dim)
        )
        if self.feedback_dim <= 0:
            raise ValueError(f"feedback_dim must be > 0, got {self.feedback_dim}")
        input_size = self.encoder_flatten_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self._init_recurrent_module(self.rnn)

        combined_weight_size = input_size + hidden_size
        self.U = nn.Parameter(torch.randn(hidden_size, self.feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(self.feedback_dim, combined_weight_size) * 0.01)
        self.proj_out = None
        if feedback_dim is not None:
            self.proj_out = nn.Linear(self.output_feedback_dim, self.feedback_dim)
        self.gate_tau = 0.5
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.register_buffer("prev_feedback", None)

        self._init_gawf_params()

    def _init_gawf_params(self) -> None:
        """
        Explicit initialization hook for GaWF-specific parameters (U and V).
        Currently preserves the initialization defined in __init__ to keep behavior unchanged.
        """
        return

    def set_feedback_frozen(self, freeze: bool):
        params = [self.U, self.V]
        if self.proj_out is not None:
            params.extend(self.proj_out.parameters())
        for p in params:
            p.requires_grad = not freeze

    def _compute_feedback(self, char_t, pos_t):
        y_t = torch.cat([char_t, pos_t], dim=-1)
        if self.proj_out is None:
            return y_t
        return self.proj_out(y_t)

    def middle_gawf(self, x_t, h_prev, fb_t):
        input_size = x_t.size(-1)
        weight_ih = self.rnn.weight_ih_l0
        weight_hh = self.rnn.weight_hh_l0
        bias_ih = self.rnn.bias_ih_l0
        bias_hh = self.rnn.bias_hh_l0
        V_ih = self.V[:, :input_size].unsqueeze(0)
        V_hh = self.V[:, input_size:].unsqueeze(0)
        trans_ih = torch.matmul(self.U, fb_t * V_ih)
        trans_hh = torch.matmul(self.U, fb_t * V_hh)
        gate_ih = torch.sigmoid(trans_ih / self.gate_tau)
        gate_hh = torch.sigmoid(trans_hh / self.gate_tau)
        gated_weight_ih = gate_ih * weight_ih.unsqueeze(0)
        gated_weight_hh = gate_hh * weight_hh.unsqueeze(0)
        ih = torch.bmm(x_t.unsqueeze(1), gated_weight_ih.transpose(1, 2)).squeeze(1)
        hh = torch.bmm(h_prev.unsqueeze(1), gated_weight_hh.transpose(1, 2)).squeeze(1)
        if bias_ih is not None:
            ih = ih + bias_ih.unsqueeze(0)
        if bias_hh is not None:
            hh = hh + bias_hh.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        gated_output = self.LNormRNN(h_t)
        gated_output = F.relu(gated_output)
        return gated_output

    def forward(self, x, use_feedback=True, reset_feedback=False):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size * frame_num, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, frame_num, -1)

        if use_feedback:
            if reset_feedback or self.prev_feedback is None:
                fb = torch.zeros(batch_size, self.feedback_dim, device=x.device, dtype=torch.float32)
            else:
                fb = self.prev_feedback.to(device=x.device, dtype=torch.float32)

            hidden_size = self.rnn.hidden_size
            char_out = torch.empty(batch_size, frame_num, self.num_classes, device=x.device, dtype=x.dtype)
            pos_out = torch.empty(batch_size, frame_num, self.num_pos, device=x.device, dtype=x.dtype)
            h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)

            for t in range(frame_num):
                x_t = x[:, t, :]
                fb_t = fb.clamp(-10, 10).unsqueeze(2)
                gated_output = self.middle_gawf(x_t, h, fb_t)
                gated_output = F.dropout(gated_output, p=self.rnn_dropout, training=self.training)
                char_t, pos_t = self.classifier(gated_output)
                if self.proj_out is None:
                    with torch.no_grad():
                        fb = self._compute_feedback(char_t, pos_t)
                else:
                    fb = self._compute_feedback(char_t, pos_t)
                h = gated_output
                char_out[:, t, :], pos_out[:, t, :] = char_t, pos_t

            self.prev_feedback = fb.detach().to(dtype=torch.float32)
        else:
            self.prev_feedback = None
            x, _ = self.rnn(x)
            x = self.LNormRNN(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.rnn_dropout, training=self.training)
            char_out, pos_out = self.classifier(x)

        return char_out, pos_out

