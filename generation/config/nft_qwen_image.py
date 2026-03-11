import imp
import os

import ml_collections

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


QWEN_IMAGE_MODEL_PATH = os.getenv("QWEN_IMAGE_MODEL_PATH", "Qwen/Qwen-Image")
QWEN_IMAGE_DATA_JSON = os.getenv("QWEN_IMAGE_DATA_JSON", "data/sharegpt/text_to_image.json")

OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "outputs")
WANDB_LOG_ROOT = os.getenv("WANDB_LOG_ROOT", "logs/wandb")


def get_config(name):
    return globals()[name]()


def _configure_grouped_batching(
    config,
    n_gpus,
    num_groups,
    num_image_per_prompt,
    gradient_step_per_epoch=1,
    train_batch_size=None,
):
    total_samples = num_groups * num_image_per_prompt
    config.sample.num_image_per_prompt = num_image_per_prompt

    if train_batch_size is not None:
        bsz = int(train_batch_size)
        if not (
            total_samples % (n_gpus * bsz) == 0
            and (n_gpus * bsz) % num_image_per_prompt == 0
        ):
            raise ValueError(
                "Invalid batch sizing: num_groups*num_image_per_prompt must be divisible by "
                "(n_gpus*train_batch_size), and (n_gpus*train_batch_size) must be divisible by num_image_per_prompt."
            )
    else:
        bsz = 4
        while bsz >= 1:
            if (
                total_samples % (n_gpus * bsz) == 0
                and (n_gpus * bsz) % num_image_per_prompt == 0
            ):
                break
            bsz -= 1
        if bsz < 1:
            raise ValueError("Cannot find valid train_batch_size for requested num_groups/num_image_per_prompt.")

    num_batches_per_epoch = total_samples // (n_gpus * bsz)
    if num_batches_per_epoch % gradient_step_per_epoch != 0:
        raise ValueError("num_batches_per_epoch must be divisible by gradient_step_per_epoch.")

    config.sample.train_batch_size = bsz
    config.sample.num_batches_per_epoch = num_batches_per_epoch
    config.train.batch_size = bsz
    config.train.gradient_accumulation_steps = num_batches_per_epoch // gradient_step_per_epoch


def _build_base_config(exp_name):
    config = base.get_config()

    n_gpus = int(os.getenv("N_GPUS", "8"))
    num_groups = int(os.getenv("NUM_GROUPS", "32"))
    num_image_per_prompt = int(os.getenv("NUM_IMAGE_PER_PROMPT", "16"))
    train_batch_size = os.getenv("TRAIN_BATCH_SIZE", "")
    train_batch_size = int(train_batch_size) if train_batch_size else 2

    config.base_model = "qwen_image"
    config.run_name = exp_name

    config.pretrained.model = QWEN_IMAGE_MODEL_PATH
    config.pretrained.revision = ""
    config.dataset = ""
    config.prompt_fn = ""

    config.dataset_json_path = QWEN_IMAGE_DATA_JSON
    config.dataset_test_ratio = 0.02
    config.dataset_split_seed = 42
    config.num_workers = 4

    config.resolution = 512
    config.sample.num_steps = 50
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4.0
    config.eval_guidance_scale = 4.0

    _configure_grouped_batching(
        config,
        n_gpus=n_gpus,
        num_groups=num_groups,
        num_image_per_prompt=num_image_per_prompt,
        gradient_step_per_epoch=1,
        train_batch_size=train_batch_size,
    )

    config.sample.test_batch_size = 4
    config.sample.global_std = True
    config.sample.deterministic = True
    config.sample.same_latent = False
    config.sample.noise_level = 1.2
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps // 2)

    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.lora = ml_collections.ConfigDict()
    config.lora.rank = 64
    config.lora.alpha = 128
    config.lora.target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    ]

    config.train.learning_rate = 1e-4
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.adv_clip_max = 5
    config.train.beta = 0.0
    config.train.adv_mode = "all"
    config.train.ema = False
    config.train.lora_path = None
    config.train.gradient_checkpointing = True

    config.beta = 1.0
    config.decay_type = 1
    config.mixed_precision = "bf16"
    config.allow_tf32 = True
    config.per_prompt_stat_tracking = True

    config.reward_fn = {"qwen_vl": 1.0}
    config.reward_client_workers = 16

    config.num_epochs = 100000000
    config.save_freq = 30
    config.eval_freq = 30

    config.wandb_mode = "offline"
    config.output_dir = f"{OUTPUT_ROOT}/{exp_name}"
    config.log_dir = f"{WANDB_LOG_ROOT}/{exp_name}"
    config.save_dir = "logs/pickscore/qwenimage"

    return config


def qwen_image_sharegpt_qwenvl():
    exp_name = os.getenv("EXP_NAME", "qwen_image_diffusionnft_sharegpt_8gpu")
    return _build_base_config(exp_name=exp_name)


def qwen_image_sharegpt_qwenvl_debug():
    config = qwen_image_sharegpt_qwenvl()
    config.debug = True
    config.num_epochs = 1
    config.sample.num_batches_per_epoch = 2
    config.train.gradient_accumulation_steps = 1
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 10
    config.save_freq = 100000000
    config.eval_freq = 100000000
    config.wandb_mode = "disabled"
    return config
