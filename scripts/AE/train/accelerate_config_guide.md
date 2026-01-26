# Accelerate Config 설정 가이드 (B200 8장)

## 설정 방법

터미널에서 다음 명령어를 실행하세요:

```bash
accelerate config
```

## 설정 옵션 (B200 8장 기준)

설정 과정에서 다음과 같이 선택하세요:

1. **In which compute environment are you running?**
   - `This machine` 선택

2. **Which type of machine are you using?**
   - `multi-GPU` 선택

3. **How many different machines will you use?**
   - `1` 입력

4. **What is the rank of this machine?**
   - `0` 입력

5. **What is the IP address of the machine that will host the main process?**
   - `localhost` 입력 (단일 머신이므로)

6. **What is the port you will use to communicate with the main process?**
   - `29500` 입력 (기본값)

7. **Are all the machines on the same local network?**
   - `yes` 선택

8. **Which GPU(s) should be used for training on this machine as a comma-seperated list?**
   - `all` 입력 (또는 `0,1,2,3,4,5,6,7`)

9. **Do you want to use DeepSpeed?**
   - `no` 선택

10. **Do you want to use FullyShardedDataParallel?**
    - `no` 선택

11. **Do you want to use Megatron-LM?**
    - `no` 선택

12. **Do you want to use PyTorch FSDP?**
    - `no` 선택

13. **Do you want to use mixed precision?**
    - `bf16` 선택 (B200은 bfloat16 지원)

## 빠른 설정 (Non-interactive)

대화형 설정이 번거롭다면, 설정 파일을 직접 생성할 수도 있습니다:

```bash
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
```

## 설정 확인

설정이 제대로 되었는지 확인:

```bash
accelerate env
```

또는

```bash
cat ~/.cache/huggingface/accelerate/default_config.yaml
```

## 예상 출력

설정이 완료되면 다음과 같은 설정 파일이 생성됩니다:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## 주의사항

- `num_processes: 8` - GPU 8개 사용
- `mixed_precision: bf16` - B200의 bfloat16 사용
- `gpu_ids: all` - 모든 GPU 사용

설정이 완료되면 `bash train_ae_onthefly.sh`를 실행하면 됩니다!
