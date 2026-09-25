# Downloaded checkpoints

Download the shared `models` folder from Google Drive and copy its **contents** into this `ckpts` directory. Keep every checkpoint directory name unchanged.

Expected layout:

```text
ckpts/
├── pi05_base_official_openpi_rlinf/          # converted official Pi0.5 base and norm stats
├── finetune_with_collectedD_ckpt/            # full 600-demo RLinf actor checkpoint
├── recap_vlm_advantage_indicator_ckpt/       # value VLM, step 3000
├── finetune_vla_advantage_TRUE_ckpt/          # all-positive LoRA adapter, step 3000
├── final_stage_finetune_ckpt/                 # final binary-advantage LoRA adapter, step 5000
├── openpi_cache/                              # offline PaliGemma SentencePiece asset
└── paligemma_tokenizer/                       # optional Hugging Face tokenizer export
```

`finetune_with_collectedD_ckpt/` is the large artifact and may be distributed as a separate download. Its empty directory is only a placeholder: before inference or later-stage fine-tuning, it must contain `model_state_dict/full_weights.pt`. Keep `dcp_checkpoint/` as well only when training must be resumable. The two later VLA checkpoints are compact adapters and require this full checkpoint as their dense base.

If the download contains the whole dated RLinf run inside `finetune_with_collectedD_ckpt/`, the root README's preparation command moves its `actor/model_state_dict/` and optional `actor/dcp_checkpoint/` into the layout above. It also updates the two adapters and the value checkpoint to reference the downloaded dense base. Run that command after placing all checkpoint folders on the destination machine.

The required runtime tokenizer is `openpi_cache/big_vision/paligemma_tokenizer.model`. The `paligemma_tokenizer/` Hugging Face export is included only for inspection and is not used by these OpenPI/RLinf workflows.
