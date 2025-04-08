"""
演示混合专家系统的脚本
该脚本会加载训练好的门控网络和两个专家模型，并在不同地形上展示效果
"""

import os
from datetime import datetime

# 首先导入isaacgym
import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

from gpugym.envs import *
from gpugym.utils import get_args, task_registry
from gpugym import LEGGED_GYM_ROOT_DIR

# 然后导入torch和其他模块
import numpy as np
import torch
import imageio
from gpu_rl.rsl_rl.modules.moe_actor_critic import MoEActorCritic
from gpu_rl.rsl_rl.modules.actor_critic import ActorCritic

def play(args):
    """
    演示混合专家系统
    
    参数:
    - args: 命令行参数
    """
    # 设置任务名称和模型路径
    args.task = "pbrs:humanoid_moe"
    
    # 如果未指定模型路径，使用默认的测试模型
    if args.load_run is None:
        args.load_run = "Mar09_12-00-00_humanoid_moe"
    
    if args.checkpoint is None:
        args.checkpoint = "model_5000.pt"
    
    # 创建环境
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 减少环境数量用于演示
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    
    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 配置演示输出
    # 创建文件夹以保存视频
    videos_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name, args.load_run)
    os.makedirs(videos_dir, exist_ok=True)
    
    # 输出信息
    print(f"Loading model from: {os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run, args.checkpoint)}")
    print(f"Videos will be saved in: {videos_dir}")
    
    # 观察和动作维度
    obs_dim = env.num_obs
    act_dim = env.num_actions
    
    # 创建MoEActorCritic模型
    policy_cfg = train_cfg.policy
    actor_critic = MoEActorCritic(
        obs_dim,
        obs_dim,
        act_dim,
        actor_hidden_dims=policy_cfg.actor_hidden_dims,
        critic_hidden_dims=policy_cfg.critic_hidden_dims,
        gating_hidden_dims=policy_cfg.gating_net_hidden_dims,
        activation=policy_cfg.activation,
        init_noise_std=policy_cfg.init_noise_std
    ).to(env.device)
    
    # 加载训练好的混合专家模型
    moe_model_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run, args.checkpoint)
    loaded_dict = torch.load(moe_model_path, map_location=env.device)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.eval()  # 评估模式
    
    # 加载行走和奔跑专家模型
    walk_expert = None
    run_expert = None
    
    if hasattr(env_cfg.env, 'experts'):
        # 加载行走专家
        if hasattr(env_cfg.env.experts, 'walk_model_path') and env_cfg.env.experts.walk_model_path:
            walk_model_path = env_cfg.env.experts.walk_model_path
            print(f"加载行走模型: {walk_model_path}")
            walk_expert = ActorCritic(obs_dim, obs_dim, act_dim).to(env.device)
            try:
                walk_checkpoint = torch.load(walk_model_path, map_location=env.device)
                walk_expert.load_state_dict(walk_checkpoint['model_state_dict'])
                walk_expert.eval()
                print("行走模型加载成功！")
            except Exception as e:
                print(f"加载行走模型失败: {e}")
                walk_expert = None
        
        # 加载奔跑专家
        if hasattr(env_cfg.env.experts, 'run_model_path') and env_cfg.env.experts.run_model_path:
            run_model_path = env_cfg.env.experts.run_model_path
            print(f"加载奔跑模型: {run_model_path}")
            run_expert = ActorCritic(obs_dim, obs_dim, act_dim).to(env.device)
            try:
                run_checkpoint = torch.load(run_model_path, map_location=env.device)
                run_expert.load_state_dict(run_checkpoint['model_state_dict'])
                run_expert.eval()
                print("奔跑模型加载成功！")
            except Exception as e:
                print(f"加载奔跑模型失败: {e}")
                run_expert = None
    
    # 如果成功加载了两个专家模型，则将它们传递给MoEActorCritic
    if walk_expert is not None and run_expert is not None:
        actor_critic.load_experts(walk_expert, run_expert)
        print("已成功加载两个专家模型到MoEActorCritic")
    else:
        print("警告：未能成功加载专家模型，MoEActorCritic将仅使用自己的actor网络")
    
    # 启用摄像机
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    camera_props.enable_tensors = True
    
    # 为每个环境创建摄像机
    cameras = []
    for i in range(env.num_envs):
        camera_handle = env.gym.create_camera_sensor(env.envs[i], camera_props)
        camera_transform = gymapi.Transform()
        camera_transform.p = gymapi.Vec3(2.0, 0.0, 1.4)  # 设置摄像机位置
        camera_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(-90))  # 设置摄像机朝向
        env.gym.set_camera_transform(camera_handle, env.envs[i], camera_transform)
        cameras.append(camera_handle)
    
    # 运行模拟
    camera_tensors = []
    frames = []
    max_steps = 2000
    selected_env_idx = 0  # 选择记录哪个环境的视频
    
    step = 0
    t0 = datetime.now()
    stop = False
    
    # 重置环境
    obs = env.reset()
    
    # 创建选择专家的历史记录
    expert_history = []
    
    # 主循环
    while not stop:
        # 模拟一帧
        with torch.no_grad():
            # 获取MoE的动作和混合权重
            actions, dominant_experts, expert_weights = actor_critic.act_with_expert_info(obs)
            expert_history.append(dominant_experts[selected_env_idx].item())
            
            # 执行动作，同时更新环境中的专家权重信息
            expert_info = (dominant_experts, expert_weights)
            obs, rewards, dones, infos = env.step(actions, expert_info)
        
        # 捕获所选环境的摄像机图像
        if len(frames) < max_steps and step % 2 == 0:  # 每2帧保存一次，减小视频大小
            env.gym.render_all_camera_sensors(env.sim)
            camera_tensor = env.gym.get_camera_image_gpu_tensor(env.sim, env.envs[selected_env_idx], cameras[selected_env_idx], gymapi.IMAGE_COLOR)
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            camera_image = torch_camera_tensor.cpu().numpy()  # [height, width, 4] RGBA
            
            # 在图像上添加专家信息
            expert_idx = dominant_experts[selected_env_idx].item()
            walk_weight = expert_weights[selected_env_idx][0].item()
            run_weight = expert_weights[selected_env_idx][1].item()
            expert_name = f"Walk({walk_weight:.2f})/Run({run_weight:.2f})"
            
            # 将RGBA转换为RGB
            rgb_image = camera_image[:, :, :3]
            
            # 将图像添加到帧列表
            frames.append(rgb_image)
            
            # 向控制台输出信息
            if step % 100 == 0:
                print(f"Step {step}, Expert: {expert_name}, Reward: {rewards[selected_env_idx].item():.2f}")
        
        # 增加步数
        step += 1
        
        # 检查是否达到最大步数
        if step >= max_steps:
            stop = True
            
    # 保存视频
    video_name = os.path.join(videos_dir, f"humanoid_moe_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    
    # 转换RGBA为RGB并保存为MP4视频
    print(f"保存视频到: {video_name}")
    imageio.mimsave(video_name, frames, fps=30)
    
    # 打印统计信息
    walk_count = expert_history.count(0)
    run_count = expert_history.count(1)
    total_count = len(expert_history)
    
    print(f"演示完成！")
    print(f"行走模型使用率: {walk_count/total_count*100:.2f}% ({walk_count}/{total_count})")
    print(f"奔跑模型使用率: {run_count/total_count*100:.2f}% ({run_count}/{total_count})")
    print(f"视频已保存至: {video_name}")
    
    env.close()

if __name__ == '__main__':
    args = get_args()
    play(args) 