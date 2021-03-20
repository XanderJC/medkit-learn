from .__head__ import *


def reverse_sequence(seqs, mask):

    batch_size, max_seq_len, dim = seqs.size()
    rev_seqs = seqs.new_zeros(seqs.size())
    seq_lens = mask.sum(axis=1)
    for b in range(batch_size):
        T = seq_lens[b].int()
        time_slice = torch.arange(T - 1, -1, -1, device=seqs.device)
        rev_seq = torch.index_select(seqs[b, :, :], 0, time_slice.to(torch.int64))
        rev_seqs[b, 0:T, :] = rev_seq

    return rev_seqs
