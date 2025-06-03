import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "ddpo-align"                # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.seed = 42                    # random seed for reproducibility.
    config.logdir = "logs"              # top-level logging directory for checkpoint saving.
    config.num_epochs = 100             # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those samples.
    config.save_freq = 20               # number of epochs between saving model checkpoints.
    config.num_checkpoint_limit = 5     # number of checkpoints to keep before overwriting old ones.
    config.mixed_precision = "fp16"     # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.allow_tf32 = True            # allow tf32 on Ampere GPUs, which can speed up training.
    config.resume_from = ""             # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value as the run that generated the saved checkpoint.
    config.use_lora = True              # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint files will also be large.

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "runwayml/stable-diffusion-v1-5"     # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.revision = "main"                            # revision of the model to load.
    
    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 50               # number of sampler inference steps.
    sample.eta = 1.0                    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0 being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.guidance_scale = 5.0         # classifier-free guidance weight. 1.0 is no guidance.
    sample.batch_size = 1               # batch size (per GPU!) to use for sampling.
    sample.num_batches_per_epoch = 2    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch * batch_size * num_gpus`.
    
    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1                # batch size (per GPU!) to use for training.
    train.use_8bit_adam = False         # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.learning_rate = 3e-4          # learning rate.
    train.adam_beta1 = 0.9              # Adam beta1.
    train.adam_beta2 = 0.999            # Adam beta2.
    train.adam_weight_decay = 1e-4      # Adam weight decay.
    train.adam_epsilon = 1e-8           # Adam epsilon.
    train.gradient_accumulation_steps = 1   # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus * gradient_accumulation_steps`.
    train.max_grad_norm = 1.0           # maximum gradient norm for gradient clipping.
    train.num_inner_epochs = 1          # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one outer epoch's round of sampling.
    train.cfg = True                    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during sampling will be used during training.
    train.adv_clip_max = 5              # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.clip_range = 1e-4             # the PPO clip range.
    train.timestep_fraction = 1.0       # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.

    ###### Prompt Function ######
    config.prompt_fn = "imagenet_animals"       # prompt function to use. see `prompts.py` for available prompt functions.
    config.prompt_fn_kwargs = {}                # kwargs to pass to the prompt function.

    ###### Reward Function ######
    config.reward_fn = "jpeg_compressibility"   # reward function to use. see `rewards.py` for available reward functions.
    
    ###### Per-Prompt Stat Tracking ######
    config.per_prompt_stat_tracking = ml_collections.ConfigDict()   # when enabled, the model will track the mean and std of reward on a per-prompt basis and use that to compute advantages. set `config.per_prompt_stat_tracking` to None to disable per-prompt stat tracking, in which case advantages will be calculated using the mean and std of the entire batch.
    config.per_prompt_stat_tracking.buffer_size = 16                # number of reward values to store in the buffer for each prompt. the buffer persists across epochs.
    config.per_prompt_stat_tracking.min_count = 16                  # the minimum number of reward values to store in the buffer before using the per-prompt mean and std. if the buffer contains fewer than `min_count` values, the mean and std of the entire batch will be used instead.
    
    return config
