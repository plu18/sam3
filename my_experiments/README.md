# SAM 3 快速启动指南 (Quick Start Guide)

这份指南记录了如何在当前环境中配置、验证并运行 SAM 3 模型。

## 1. 准备工作 (Prerequisites)

在开始之前，请确保你已经完成了以下步骤：

1.  **注册 Hugging Face 账号**: 如果还没有，请前往 [huggingface.co](https://huggingface.co/) 注册。
2.  **申请模型访问权限**: SAM 3 模型是受限访问的。
    *   访问 [facebook/sam3 模型页面](https://huggingface.co/facebook/sam3)。
    *   点击页面上的申请/同意按钮，签署用户协议。
    *   **等待批准**（通常很快，但也可能需要一些时间）。
3.  **获取 Access Token**:
    *   访问 [Settings > Tokens](https://huggingface.co/settings/tokens)。
    *   创建一个新的 Token (类型选择 "Read")。
    *   复制这个 Token。

## 2. 环境认证 (Authentication)

我们需要在环境中登录 Hugging Face，以便脚本能自动下载模型权重。

运行我们准备好的认证脚本：

```bash
python my_experiments/auth_hf.py
```

*   脚本运行后，会提示你输入 Token。
*   粘贴你刚才复制的 Token 并回车。
*   脚本会自动检查你是否登录成功，以及是否有权限访问 `facebook/sam3`。

## 3. 运行推理测试 (Running Inference)

一旦认证成功且获得了模型访问权限，就可以运行推理测试脚本了。

```bash
python my_experiments/inference_test.py
```

**脚本功能：**
1.  自动从 Hugging Face 下载 SAM 3 模型权重 (`sam3.pt`)。
2.  生成一张红色的测试图片 (`test_image.jpg`)。
3.  加载模型并对图片进行推理（提示词："a red square"）。
4.  将结果保存到 `inference_results.txt`。

## 4. 常见问题 (Troubleshooting)

*   **403 Client Error / Cannot access gated repo**:
    *   **原因 1**: 你还没有获得 `facebook/sam3` 的访问权限。
        *   解决：请再次确认你是否在 [facebook/sam3](https://huggingface.co/facebook/sam3) 网页上点击了同意协议。
    *   **原因 2 (Fine-grained Token)**: 如果你使用的是 Fine-grained Token，可能缺少了访问 Gated Repos 的权限。
        *   解决：在创建或编辑 Token 时，请确保在 **"Repository permissions"** 中勾选了 **"Read access to contents of all public gated repositories"** (或者类似的 Gated Repos 选项)。
        *   **推荐**：为了简单起见，你可以创建一个 **"Classic"** 类型的 Token (Access Tokens -> Create new token -> Type: Write/Read)，通常默认包含所需权限。

*   **ImportError / ModuleNotFoundError**:
    *   原因：环境依赖可能未安装完全。
    *   解决：确保你是在 DevContainer 环境中运行。如果缺少 `huggingface_hub`，可以运行 `pip install huggingface_hub`。

## 文件说明

*   `my_experiments/auth_hf.py`: 用于辅助 Hugging Face 登录和权限检查的脚本。
*   `my_experiments/inference_test.py`: 用于验证模型加载和推理流程的测试脚本。
*   `my_experiments/inference_results.txt`: 推理运行成功后生成的日志文件。

