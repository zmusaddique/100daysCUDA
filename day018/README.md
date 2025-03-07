### Log

#### Think log (!Deepseek)

I got a thought this moring, how am I getting dO?

Fixes:

- Wrongly iterating over q_tile instead of kv_tile
- In test code and cuda, I need to maintain the same equations

Still buggy,

Wait let's output dQ and check.

dQ_manual:
tensor([[[[-0.0137,  0.1686,  0.1254,  0.1340],
          [ 0.3367, -0.9961,  0.1930, -0.3312],
          [ 0.0345,  0.6728, -0.5606, -0.0312],
          [-0.0745, -0.0756,  0.0784,  0.0027]]]], device='cuda:0',
grad_fn=<MulBackward0>)

dQ_cuda:
tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0233, -0.0736,  0.2137,  0.0987],
          [ 0.0286,  0.6309, -0.5214, -0.0293],
          [-0.0745, -0.0756,  0.0784,  0.0027]]]]

Every time the top row is empty. What variables is dQ dependent on? dS and K. K is given whereas dS is computed. Let's check if dS is correctly computed.

dS too differs. It is the case only with causal attention. Perhaps I shouldn't apply the mask

AssertionError: dS differs! dS_manual:
tensor([[[[-1.8708e-08, -1.3340e+00,  1.3969e-01,  1.5172e-01],
          [ 1.9902e-01, -1.9902e-01, -1.7371e+00, -2.3938e-01],
          [-4.7315e-02,  4.5658e-02,  1.6575e-03, -5.4871e-01],
          [ 1.1068e-01,  5.8861e-02,  2.8708e-02, -1.9824e-01]]]],
device='cuda:0', grad_fn=<MulBackward0>)
dS_manual.shape: torch.Size([1, 1, 4, 4])

dS_cuda:
tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.4662, -0.4662,  0.0000,  0.0000],
          [-0.0587,  0.0566,  0.0021,  0.0000],
          [ 0.1107,  0.0589,  0.0287, -0.1982]]]], device='cuda:0')
dS_cuda.shape: torch.Size([1, 1, 4, 4])

---

dS_manual(causal=False) :
tensor([[[[ 0.0722, -0.2254,  0.0252,  0.1281],
          [-0.0661, -0.0298,  0.3251, -0.2292],
          [ 0.2057,  0.4506, -0.3068, -0.3496],
          [-0.1170,  0.1011, -0.3622,  0.3781]]]], device='cuda:0',
grad_fn=<MulBackward0>

dS_manual (causal=True):
tensor([[[[ 0.0000, -0.2867,  0.0100, -0.3387],
          [-0.0140,  0.0140,  0.5775, -0.1110],
          [ 0.0719,  0.3331, -0.4050, -0.5126],
          [-0.1170,  0.1011, -0.3622,  0.3781]]]], device='cuda:0',
grad_fn=<MulBackward0>)

But how does it look for non-causal? It is different form causal. Then how exactly is dS computed for causal? I don't think the mask for causal should be applied in backward pass which appears on dS_cuda. What is dS dependent on? it is P & dP. Current understanding says, causal masking is applied only to S (QK^T). Is it applied to S? I believe there is masking at multiple places. I backward pass I get P, not S. P is just softmax applied to P. masked values will remain masked. I need to ensure only P is masked.

An hour of debugging later:
Guess what could go wrong? IT WAS THE **TEST CASE!!!!**. There was a mismatch in the function signature `manual_attention_backward(dO, Q, K, V, O, M, P, causal)`. This should have been easily been caught my the LSP but not sure why it didn't. More importantly I believe the variable P must have been shodowed which is now fixed.

# Verdict: dQ is computed successfully!

will resume correction of dk & dV tom
