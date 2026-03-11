import imp
import os

import ml_collections

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


ZIMAGE_MODEL_ID_WITH_ORIGIN_PATHS = os.getenv(
    "ZIMAGE_MODEL_ID_WITH_ORIGIN_PATHS",
    "models/Z-Image-Turbo:transformer/*.safetensors,"
    "models/Z-Image-Turbo:text_encoder/*.safetensors,"
    "models/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors",
)
ZIMAGE_MODEL_PATHS = os.getenv("ZIMAGE_MODEL_PATHS", "")
ZIMAGE_TOKENIZER_PATH = os.getenv(
    "ZIMAGE_TOKENIZER_PATH", "models/Z-Image-Turbo/tokenizer"
)
ZIMAGE_DATA_JSON = os.getenv(
    "ZIMAGE_DATA_JSON",
    "data/sharegpt/text_to_image.json",
)

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
    config.num_groups = num_groups

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
        bsz = 9
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
    num_groups = int(os.getenv("NUM_GROUPS", "48"))
    num_image_per_prompt = int(os.getenv("NUM_IMAGE_PER_PROMPT", "12"))

    config.base_model = "zimage_turbo"
    config.run_name = exp_name

    config.pretrained.model_id_with_origin_paths = ZIMAGE_MODEL_ID_WITH_ORIGIN_PATHS
    config.pretrained.model_paths = ZIMAGE_MODEL_PATHS
    config.pretrained.tokenizer_path = ZIMAGE_TOKENIZER_PATH
    config.pretrained.fp8_models = ""
    config.pretrained.offload_models = ""

    config.dataset_json_path = ZIMAGE_DATA_JSON
    config.dataset_test_ratio = 0.02
    config.dataset_split_seed = 42
    config.num_workers = 4

    config.resolution = 512
    config.sample.num_steps = 8
    config.sample.eval_num_steps = 16
    config.sample.guidance_scale = 1.0
    config.train_guidance_scale = 1.0
    config.eval_guidance_scale = 1.0
    config.sample.sigma_shift = 3.0
    config.sample.eval_sigma_shift = 3.0
    config.sample.denoising_strength = 1.0
    config.sample.eval_denoising_strength = 1.0
    _configure_grouped_batching(
        config,
        n_gpus=n_gpus,
        num_groups=num_groups,
        num_image_per_prompt=num_image_per_prompt,
        gradient_step_per_epoch=1,
    )
    config.sample.test_batch_size = 16
    config.sample.global_std = True
    config.sample.deterministic = True

    config.use_lora = True
    config.lora = ml_collections.ConfigDict()
    config.lora.rank = 32
    config.lora.alpha = 32
    config.lora.target_modules = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]

    config.train.learning_rate = 1e-4
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.adv_clip_max = 5
    config.train.beta = 0.001
    config.train.adv_mode = "positive_only"
    config.train.policy_loss_scale = 0.5
    config.train.ema = True
    config.train.lora_path = None
    config.train.gradient_checkpointing = True
    config.train.gradient_checkpointing_offload = False

    config.beta = 0.2
    config.decay_type = 1
    config.mixed_precision = "bf16"
    config.model_load_dtype = "bf16"
    config.allow_tf32 = True
    config.per_prompt_stat_tracking = True
    config.enable_npu_patch = False

    config.reward_fn = {"qwen_vl": 1.0}
    config.reward_client_workers = 8
    config.reward_max_pending = 1

    config.num_epochs = 100000
    config.save_freq = 150
    config.eval_freq = 1000
    config.eval_steps = 1000
    config.eval_num_prompts = 64

    config.wandb_mode = "offline"
    config.output_dir = f"{OUTPUT_ROOT}/{exp_name}"
    config.log_dir = f"{WANDB_LOG_ROOT}/{exp_name}"
    config.save_dir = config.output_dir

    return config


def zimage_turbo_sharegpt_qwenvl():
    exp_name = os.getenv("EXP_NAME", "zimage_turbo_diffusionnft_sharegpt_8gpu")
    config = _build_base_config(exp_name=exp_name)
    return config


def zimage_turbo_sharegpt_qwenvl_debug():
    config = zimage_turbo_sharegpt_qwenvl()
    config.debug = True
    config.num_epochs = 1
    config.sample.num_batches_per_epoch = 2
    config.train.gradient_accumulation_steps = 1
    config.sample.num_steps = 4
    config.sample.eval_num_steps = 8
    config.eval_steps = 10000
    config.save_freq = 50
    config.eval_freq = 100000
    config.wandb_mode = "disabled"
    return config
