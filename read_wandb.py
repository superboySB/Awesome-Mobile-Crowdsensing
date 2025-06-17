from wandb import Api
import pandas as pd


def get_metric_history(run_path, metric_name, api_key=None):
    """
    从指定的 W&B run 中获取某个指标的完整历史记录
    
    参数:
        run_path: str, 格式为 "entity/project/run_id"
        metric_name: str, 要获取的指标名称
        api_key: str, 可选, W&B API密钥
    
    返回:
        list: 包含该指标所有非None记录值的列表
    """
    api = Api(api_key=api_key)
    run = api.run(run_path)

    # 获取历史记录
    history = run.scan_history()

    # 提取指定指标的值并过滤掉None值
    metric_values = [row[metric_name] for row in history if metric_name in row and row[metric_name] is not None]

    return metric_values


# 使用示例
if __name__ == "__main__":
    # RUN_PATH = "aequatio/awesome-mcs/3mz40utj"
    # METRIC_NAME = "valid_handling_index"
    RUN_PATH = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
    METRIC_NAME = "valid_handling_index"
    values = get_metric_history(RUN_PATH, METRIC_NAME, API_KEY)
    print(f"获取到 {len(values)} 个有效的 {METRIC_NAME} 值")
    print(f"前5个值: {values[:5]}")
