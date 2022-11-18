import torch
import torch.nn as nn
import numpy as np

from torch.distributions.normal import Normal

# 1 Layered classifier
# Input Vector -> Scores for each candidate 
class L4_Tower(nn.Module):
    def __init__(self, hidden_size):
        super(L4_Tower, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        return self.sigmoid(out)

# 2 Layered MLP
# Input Vector -> hidden_size Vector
class L3_Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(L3_Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



class MoE(nn.Module):
    """Gated MoE where each expert is a 2-layer Feed-Forward networkss.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, device, n_metrics, input_size, output_size, num_experts, hidden_size, k=4):
        super(MoE, self).__init__()
        self.device = device
        self.n_metrics = n_metrics
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.softmax = nn.Softmax(1)
        # instantiate experts
        self.experts = nn.ModuleList([L3_Expert(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        #Gates for Experts
        self.w_gate = nn.ParameterList([nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True) for i in range(n_metrics)])
        #Noise to add some randomness while picking topk gates
        self.w_noise = nn.ParameterList([nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True) for i in range(n_metrics)])
        self.normal = Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device))
        self.softplus = nn.Softplus()

    def regularisation_allExpertsUtilisation(self, x):
        """We will be using Coefficient of variation https://en.wikipedia.org/wiki/Coefficient_of_variation
        to push the variation of  "Number of samples going to each expert" to zero 
        i.e to promote all experts to be utilised instead of model using only a few experts for all tasks
        """
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + 1e-8)

    def noisy_top_k_gating(self, gate_idx, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate[gate_idx]
        if train:
            raw_noise_stddev = x @ self.w_noise[gate_idx]
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # if train and self.k < self.num_experts:
        #     thresh = top_logits[:, self.k]
        #     print(clean_logits.shape,thresh.shape,noise_stddev.shape)
        #     load = self.normal.cdf((clean_logits - thresh)/noise_stddev).sum(0)
        # else:
        #     load = (gates > 0).sum(0)
        load = (gates > 0).sum(0)

        return gates, load

    def forward(self, x, train=True, collect_gates = False, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        all_y = []
        final_loss = torch.tensor(0.0).to(self.device)
        for gate_idx in range(self.n_metrics):
            sample_experts_importance, load = self.noisy_top_k_gating(gate_idx, x, train)
            # calculate importance loss
            experts_importance = sample_experts_importance.sum(0)

            loss = self.regularisation_allExpertsUtilisation(experts_importance) + self.regularisation_allExpertsUtilisation(load)
            loss *= loss_coef


            nonzero_experts = torch.nonzero(sample_experts_importance)
            batchIndexes_sortedbyExpert = nonzero_experts[nonzero_experts[:,1].sort()[1]][:,0]
            numSamples_EachExpert = list((sample_experts_importance > 0).sum(0).detach().cpu().numpy())
            # print(nonzero_experts[nonzero_experts[:,1].sort(0)[1]].shape)
            experts_inds_=nonzero_experts[nonzero_experts[:,1].sort(0)[1]]
            sample_experts_importance_Expertwise = (sample_experts_importance[experts_inds_[:,0],experts_inds_[:,1]]).unsqueeze(1)
            # sample_experts_importance_Expertwise = sample_experts_importance_Expertwise.flatten()
            expert_inputs = torch.split(x[batchIndexes_sortedbyExpert].squeeze(1), numSamples_EachExpert, dim=0)
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

            # apply exp to expert outputs, so we are not longer in log space
            expert_outputs_tens = torch.cat(expert_outputs, 0).exp()

            expert_outputs_tens = expert_outputs_tens.mul(sample_experts_importance_Expertwise)
            zeros = torch.zeros(x.size(0), expert_outputs[-1].size(1), requires_grad=True).to(self.device)
            # combine samples that have been processed by the same k experts


            combined = zeros.index_add(0, batchIndexes_sortedbyExpert, expert_outputs_tens.float())
            # add eps to all zero values in order to avoid nans when going back to log space
            combined[combined == 0] = np.finfo(float).eps
            # back to log space
            y = combined.log()

            all_y.append(y)
            final_loss = final_loss + loss

        return all_y, final_loss


