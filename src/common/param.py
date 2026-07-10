import argparse
import os
import datetime
from pathlib import Path
from utils.CN import CN
import transformers
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CommonArguments:
    project_prefix: str = field(
        default_factory=lambda: str(Path(str(os.getcwd())).parent.resolve()),
        metadata={"help": "project path"}
    )
    run_type: str = field(default="train", metadata={"help": "run_type in [collect, train, eval]"})
    collect_type: str = field(default="dagger", metadata={"help": "collect_type in [dagger]"})
    name: str = field(default='default', metadata={"help": 'experiment name'})

    maxInput: int = field(default=500, metadata={"help": "max input instruction"})
    maxWaypoints: int = field(default=500, metadata={"help": 'max action sequence'})
    enable_comm_delay: bool = field(default=True, metadata={"help": "inject uplink delay in continuous eval"})
    comm_trace_csv_path: Optional[str] = field(default=None, metadata={"help": "bandwidth trace csv path"})
    chunk_waypoints: int = field(default=1, metadata={"help": "continuous eval decision interval in executed waypoints"})
    edge_vlm_host: str = field(default="127.0.0.1", metadata={"help": "edge VLM server host"})
    edge_vlm_bind_host: str = field(default="0.0.0.0", metadata={"help": "edge VLM server bind host"})
    edge_vlm_port: int = field(default=26000, metadata={"help": "edge VLM server port"})
    coarse_goal_min_distance: float = field(default=0.5, metadata={"help": "deprecated; accepted for old edge DNN scripts but no longer used"})
    scheduler_model_path: Optional[str] = field(default=None, metadata={"help": "PPO scheduler model path for DRL scheduler eval"})
    scheduler_total_episodes: int = field(default=200, metadata={"help": "total episodes for DRL scheduler training (primary stopping criterion)"})
    scheduler_total_timesteps: int = field(default=100000, metadata={"help": "max PPO timesteps for DRL scheduler training (safety cap)"})
    scheduler_max_steps: int = field(default=800, metadata={"help": "max scheduler decisions per episode for DRL scheduler"})
    scheduler_learning_rate: float = field(default=0.0003, metadata={"help": "PPO learning rate for DRL scheduler"})
    scheduler_n_steps: int = field(default=64, metadata={"help": "PPO rollout n_steps for DRL scheduler"})
    scheduler_batch_size: int = field(default=32, metadata={"help": "PPO batch size for DRL scheduler"})
    scheduler_gamma: float = field(default=0.99, metadata={"help": "PPO discount factor for DRL scheduler"})
    scheduler_time_norm_ms: float = field(default=5000.0, metadata={"help": "reward normalization T0 for scheduler elapsed time"})
    scheduler_drift_norm_m: float = field(default=2.5, metadata={"help": "reward normalization delta0 for scheduler state drift"})
    scheduler_time_drift_norm_ms: float = field(default=5000.0, metadata={"help": "reward normalization tau0 for scheduler time drift increase"})
    scheduler_ne_norm_m: float = field(default=1.0, metadata={"help": "reward normalization NE0 for navigation error progress"})
    scheduler_ne_progress_weight: float = field(default=1.0, metadata={"help": "reward weight for navigation error progress"})
    scheduler_time_weight: float = field(default=1.0, metadata={"help": "reward weight for scheduler elapsed time"})
    scheduler_drift_weight: float = field(default=1.0, metadata={"help": "reward weight for scheduler state drift increase"})
    scheduler_time_drift_weight: float = field(default=1.0, metadata={"help": "reward weight for scheduler time drift increase"})
    scheduler_request_weight: float = field(default=0.05, metadata={"help": "reward penalty for edge VLM request"})
    scheduler_success_reward: float = field(default=40.0, metadata={"help": "terminal reward for SR navigation success"})
    scheduler_oracle_success_reward: float = field(default=20.0, metadata={"help": "terminal reward for OSR-only navigation success"})
    scheduler_collision_penalty: float = field(default=20.0, metadata={"help": "terminal penalty for collision"})
    scheduler_failure_penalty: float = field(default=10.0, metadata={"help": "terminal penalty for failure or maxWaypoints"})
    scheduler_illegal_action_penalty: float = field(default=5.0, metadata={"help": "penalty for choosing an illegal action in the current state"})
    scheduler_idle_wait_ms: float = field(default=100.0, metadata={"help": "small hover duration for STOP_NO_REQUEST when no request is in flight"})

    dagger_it: int = field(default=1)
    epochs: int = field(default=10)
    lr: float = field(default=0.00025, metadata={"help": "learning rate"})
    batchSize: int = field(default=8)
    trainer_gpu_device: int = field(default=0, metadata={"help": 'GPU'})

    inflection_weight_coef: float = field(default=1.9)

    dagger_mode_load_scene: List[str] = field(default_factory=list)
    dagger_update_size: int = field(default=8000)
    dagger_mode: str = field(default="end", metadata={"help": 'dagger mode in [end middle nearest]'})
    dagger_p: float = field(default=1.0, metadata={"help": 'dagger p'})

    tokenizer_use_bert: bool = field(default=True)

    simulator_tool_port: int = field(default=30000, metadata={"help": "simulator_tool port"})
    DDP_MASTER_PORT: int = field(default=20001, metadata={"help": "DDP MASTER_PORT"})

    continue_start_from_dagger_it: Optional[int] = field(default=None)
    continue_start_from_checkpoint_path: Optional[str] = field(default=None)

    vlnbert: bool = field(default=False)
    featdropout: float = field(default=0.4)
    action_feature: int = field(default=32)
    
    eval_save_path: Optional[str] = field(default=None)
    dagger_save_path: Optional[str] = field(default=None)
    activate_maps: Optional[List[str]] = field(default_factory=list)

    gpu_id: int = field(default=3, metadata={"help": "simulator gpus"})
    always_help: bool = field(default=False)
    use_gt: bool = field(default=False)

    dataset_path: Optional[str] = field(default=None)
    eval_json_path: Optional[str] = field(default=None)
    train_json_path: Optional[str] = field(default=None)
    object_name_json_path: Optional[str] = field(default=None)
    map_spawn_area_json_path: Optional[str] = field(default=None)
    max_episodes_per_scene: int = field(default=5000, metadata={"help": "max episodes per scene before forced UE4 restart"})
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=True)
    mm_use_im_start_end: bool = field(default=False)

    
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="facebook/opt-350m")
    model_base: Optional[str] = field(default=None)
    traj_model_path: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    groundingdino_config: Optional[str] = field(default=None)
    groundingdino_model_path: Optional[str] = field(default=None)
    
    
parser = transformers.HfArgumentParser((CommonArguments, ModelArguments, DataArguments))
args, model_args, data_args = parser.parse_args_into_dataclasses()

args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = '{}/workdir/{}/logs/{}_{}.log'.format(args.project_prefix, args.run_type, args.collect_type, args.make_dir_time)


# args.run_type = 'collect'
assert args.run_type in ['collect', 'train', 'eval'], 'run_type error'
# args.collect_type = 'TF'
assert args.collect_type in ['TF', 'dagger'], 'collect_type error'


args.machines_info = [
    {
        'MACHINE_IP': '127.0.0.1',
        'SOCKET_PORT': int(args.simulator_tool_port),
        'MAX_SCENE_NUM': 16,
        'open_scenes': [],
    },
]


args.TRAIN_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.TRAINVAL_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.vocab_size = 10038


default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()
