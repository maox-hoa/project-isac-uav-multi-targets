import torch, time

N = 6000

A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)

t0 = time.time()
_ = A_cpu @ B_cpu
print("CPU time:", time.time() - t0)

A_gpu = A_cpu.cuda()
B_gpu = B_cpu.cuda()

torch.cuda.synchronize()
t0 = time.time()
_ = A_gpu @ B_gpu
torch.cuda.synchronize()
print("GPU time:", time.time() - t0)