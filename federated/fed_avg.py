import torch


def fed_avg(global_model, client_params_list, client_weights_list):
    """
    联邦平均算法（FedAvg）

    参数:
        global_model: 全局模型（用于获取参数结构）
        client_params_list: 列表，每个元素是客户端的模型参数列表（即 get_parameters() 返回的格式）
        client_weights_list: 列表，每个元素对应客户端的本地样本数量（用于加权）

    返回:
        new_params: 聚合后的全局模型参数列表（与 client_params_list[0] 结构相同）
    """
    # 计算总样本数
    total_samples = sum(client_weights_list)

    # 初始化新参数为全零（与第一个客户端的参数结构相同）
    new_params = [torch.zeros_like(param) for param in client_params_list[0]]

    # 加权求和
    for params, weight in zip(client_params_list, client_weights_list):
        for i, param in enumerate(params):
            new_params[i] += (weight / total_samples) * param

    return new_params


# 简单测试（直接运行本文件时执行）
if __name__ == '__main__':
    # 创建两个模拟参数列表（每个参数是一个简单的张量）
    params1 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    params2 = [torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])]

    # 模拟样本权重：客户端1有100个样本，客户端2有300个样本
    weights = [100, 300]

    # 聚合
    aggregated = fed_avg(None, [params1, params2], weights)

    # 打印结果
    print("聚合后的参数：")
    for i, p in enumerate(aggregated):
        print(f"参数{i + 1}: {p.tolist()}")

    # 手动验证：加权平均应为 (100*params1 + 300*params2)/400
    expected1 = (100 * torch.tensor([1.0, 2.0]) + 300 * torch.tensor([5.0, 6.0])) / 400
    expected2 = (100 * torch.tensor([3.0, 4.0]) + 300 * torch.tensor([7.0, 8.0])) / 400
    print("\n期望结果：")
    print(f"参数1: {expected1.tolist()}")
    print(f"参数2: {expected2.tolist()}")