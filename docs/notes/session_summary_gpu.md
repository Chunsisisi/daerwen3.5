文档状态：`REFERENCE`

本次会话摘要（WSL GPU & 性能测试）
=================================

时间线简述
--------
- 在 `daerwen3.5/.venv` 安装了 `cupy-cuda12x` 13.6.0。
- 执行 CPU 性能测试（70×70，400 粒子，solar=0.6，无守护）：
  - 150 步总耗时约 17.1s，粒子数从 400 爆增到 ~4.85 万，步时随粒子数指数增长。
  - 更长试跑 132 步时粒子数达 10.6 万，30s 超时提前停止。
- 尝试开启 GPU：安装 `nvidia-cuda-toolkit`（约 7GB，完整安装成功）。
- 安装后在 WSL 内用 `.venv` + `cupy-cuda12x` 测试仍报 `cudaErrorOperatingSystem: OS call failed or operation not supported on this OS`，`LD_LIBRARY_PATH=/usr/lib/wsl/lib` 也无效。
- WSL 内存在 `/dev/dxg`，但无 `/dev/nvidia*`；`nvidia-smi` 可用（透传），说明驱动可读但 CUDA runtime 仍未连通。

待办/操作指南
-----------
1) 在 **Windows 侧**执行 `wsl --shutdown` 重启 WSL，使新装 CUDA 库生效；确认发行版是 WSL2（`wsl -l -v`，必要时 `wsl --set-version <发行版> 2`）。
2) 重启后在 WSL 中验证 GPU：
   ```
   cd /mnt/f/avalanche-持续学习/daerwen3.5
   . .venv/bin/activate
   LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_VISIBLE_DEVICES=0 python - <<'PY'
   import cupy as cp
   print(cp.__version__)
   print(cp.cuda.runtime.getDeviceCount())
   PY
   ```
   若仍报同样错误，检查 Windows 驱动是否为支持 WSL 的版本，或 `.wslconfig` 是否禁用了 GPU。
3) 需要 GPU 时可继续使用现有 `cupy-cuda12x`；等待官方 CUDA 13 wheel 发布后再升级。

笔记
----
- 综合测试套件很耗时（预热 1000 步 + 多轮扰动），默认 CPU 跑需数分钟；短期性能基线可用上面的 150 步脚本。
- 粒子爆炸的主要瓶颈在 Python 层循环（代谢/交互/复制/移动）；后续可通过 Numba/CuPy 内核化降低步时。
