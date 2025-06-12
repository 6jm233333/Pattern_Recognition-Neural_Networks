import torch
import platform
import subprocess

def print_system_info():
    """打印系统和硬件信息"""
    print("\n" + "="*50)
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"处理器: {platform.processor()}")

def check_gpu_info():
    """检查NVIDIA GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        print("\nNVIDIA GPU信息:")
        print(result.stdout.decode('utf-8'))
    except FileNotFoundError:
        print("\n警告: 未找到nvidia-smi命令，可能没有安装NVIDIA驱动")

def verify_pytorch():
    """验证PyTorch安装和功能"""
    print("\n" + "="*50)
    print("PyTorch验证:")
    
    # 基本版本信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # CPU功能测试
    print("\nCPU功能测试:")
    try:
        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = a + b
        print(f"张量加法测试成功: \n{a} + \n{b} = \n{c}")
        print(f"张量形状: {c.shape}")
    except Exception as e:
        print(f"CPU测试失败: {str(e)}")
        return False
    
    # GPU功能测试
    if torch.cuda.is_available():
        print("\nGPU功能测试:")
        try:
            device = torch.device("cuda:0")
            print(f"检测到GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            
            # 内存测试
            print(f"\nGPU内存信息:")
            print(f"分配内存: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
            print(f"保留内存: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
            
            # 计算测试
            x = torch.rand(1000, 1000, device=device)
            y = torch.rand(1000, 1000, device=device)
            z = torch.mm(x, y)
            print(f"\n矩阵乘法测试成功 (结果形状: {z.shape})")
            
            # 性能测试
            from time import time
            start_time = time()
            for _ in range(100):
                torch.mm(x, y)
            elapsed = time() - start_time
            print(f"100次矩阵乘法耗时: {elapsed:.4f}秒")
            
        except Exception as e:
            print(f"GPU测试失败: {str(e)}")
            return False
    else:
        print("\n警告: 未检测到GPU支持")
        print("可能原因:")
        print("1. 没有安装NVIDIA驱动")
        print("2. CUDA工具包未安装或版本不匹配")
        print("3. PyTorch安装的不是GPU版本")
    
    return True

def main():
    print("="*50)
    print("PyTorch完整验证脚本")
    print("="*50)
    
    print_system_info()
    check_gpu_info()
    
    if verify_pytorch():
        print("\n" + "="*50)
        print("✅ PyTorch验证通过，所有测试成功！")
    else:
        print("\n" + "="*50)
        print("❌ PyTorch验证失败，请检查上述错误信息")
    
    print("\n提示: 如果遇到GPU相关问题，请检查:")
    print("1. NVIDIA驱动是否安装 (nvidia-smi)")
    print("2. CUDA工具包版本是否匹配")
    print("3. PyTorch是否安装了GPU版本")

if __name__ == "__main__":
    main()