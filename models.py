import os
import csv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    # transformers 4/5 Cache objects: ask the object (subscripting was removed in v5)
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "layers"):
        layers = past_key_values.layers
        return int(layers[0].keys.shape[-2]) if layers else 0
    if not past_key_values:                       # empty legacy tuple / None-ish
        return 0
    return int(past_key_values[0][0].shape[-2])   # legacy tuple of (K, V)


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        if torch.cuda.is_available():
            # bf16 needs Ampere+ (L4/A10/A100/3090/4090); a T4 is Turing -> fp16.
            load_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            load_dtype = torch.float32
        # low_cpu_mem_usage avoids the ~2x-model-size RAM spike during loading that
        # SIGKILLs (-9) a 4B model on Colab's ~12.7GB RAM. The print surfaces whether
        # CUDA is actually active -- fp32 on a CPU runtime is the other way to OOM.
        print(f"[ModelWrapper] loading {model_name} | cuda={torch.cuda.is_available()} "
              f"| dtype={load_dtype} | device={device}", flush=True)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=load_dtype, low_cpu_mem_usage=True,
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True,
                    enable_thinking: bool = False) -> str:
        # Default OFF: short-answer QA (HotpotQA/GSM8K) doesn't need text <think>,
        # and thinking blocks eat the generation budget before the answer is emitted.
        # The LATENT reasoning (latent_steps) is a separate mechanism and is unaffected.
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                )
            except TypeError:   # template that doesn't accept the kwarg
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,   # default OFF — see render_chat
        padding_side: Optional[str] = None,  # "left" for batched decode-from-prefix
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(
                messages, add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking))
        saved_side = self.tokenizer.padding_side
        if padding_side is not None:
            self.tokenizer.padding_side = padding_side
        try:
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = saved_side
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    @torch.no_grad()
    def decode_from_cache(self, input_ids, past_key_values, max_new_tokens: int = 64,
                          eos_id: Optional[int] = None) -> str:
        """Greedy-decode `input_ids` on top of a prefilled cache, via manual forwards.

        HF `generate()` assumes past_key_values is a prefix of the ongoing
        sequence; here the cache is SEPARATE content (the workers' states) that
        the judge attends to but which is not a literal prefix of the judge
        prompt -- so generate()'s bookkeeping breaks. This loop makes no such
        assumption: it just forwards the judge tokens over the given cache, with
        positions continuing from the cache length. (Same reason LatentMAS
        bypasses generate() in its latent loop.)
        """
        hf = getattr(self, "HF_model", None) or self.model
        dev = self.device
        ids = input_ids.to(dev)
        if eos_id is None:
            eos_id = self.tokenizer.eos_token_id

        past = past_key_values
        plen = _past_length(past)
        pos = torch.arange(plen, plen + ids.shape[-1], device=dev).unsqueeze(0)
        attn = torch.ones(1, plen + ids.shape[-1], dtype=torch.long, device=dev)
        out = hf(input_ids=ids, attention_mask=attn, position_ids=pos,
                 past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)

        gen = []
        for _ in range(max_new_tokens):
            if eos_id is not None and int(nxt.item()) == eos_id:
                break
            gen.append(nxt)
            plen = _past_length(past)
            pos = torch.tensor([[plen]], device=dev)
            attn = torch.ones(1, plen + 1, dtype=torch.long, device=dev)
            out = hf(input_ids=nxt, attention_mask=attn, position_ids=pos,
                     past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)

        if not gen:
            return ""
        return self.tokenizer.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)

    @torch.no_grad()
    def decode_text_batch_from_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> List[str]:
        """Decode N sequences IN ONE BATCH, each continuing from the same shared
        prefix cache (or from nothing when past_key_values is None).

        This is the routed_nl fan-out: the workers are a genuine batched forward
        pass -- one GPU call per decode step for all workers -- not a Python loop
        over workers. `input_ids`/`attention_mask` must be LEFT-padded
        (prepare_chat_batch(..., padding_side="left")); the shared prefix must be
        batch-N already (cache_ops.expand_cache).

        Positions are computed PER ROW from the attention mask, so each row's
        real tokens sit at prefix_len, prefix_len+1, ... regardless of how much
        padding sits to its left -- we do not trust generate()'s uniform
        cache_position bookkeeping with a non-uniform batch. Pad slots stay
        masked for the whole decode, so their K/V are never attended to.

        temperature <= 0 means greedy; otherwise temperature/top_p sampling
        (matches the repo's sampled-judge convention).
        """
        hf = getattr(self, "HF_model", None) or self.model
        dev = self.device
        ids = input_ids.to(dev)
        mask = attention_mask.to(dev)
        n, seq_len = ids.shape
        plen = _past_length(past_key_values)

        # Row-wise positions: pads before a row's first real token are masked out;
        # real token j of a row sits at plen + j.
        pos = mask.long().cumsum(-1) - 1 + plen
        pos = pos.clamp_min(0)
        full_mask = mask
        if plen > 0:
            past_mask = torch.ones((n, plen), dtype=mask.dtype, device=dev)
            full_mask = torch.cat([past_mask, mask], dim=-1)

        out = hf(input_ids=ids, attention_mask=full_mask, position_ids=pos,
                 past_key_values=past_key_values, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        next_pos = (pos.max(dim=-1).values + 1)                # [n] per-row next position
        finished = torch.zeros(n, dtype=torch.bool, device=dev)
        gen_cols: List[torch.Tensor] = []

        def _pick(lg: torch.Tensor) -> torch.Tensor:
            if temperature is None or temperature <= 0:
                return lg.argmax(-1, keepdim=True)
            probs = torch.softmax(lg / temperature, dim=-1)
            if top_p is not None and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True, dim=-1)
                cum = sp.cumsum(-1)
                cut = cum - sp >= top_p                        # tokens beyond the nucleus
                sp = sp.masked_fill(cut, 0.0)
                sp = sp / sp.sum(-1, keepdim=True)
                choice = torch.multinomial(sp, 1)
                return si.gather(-1, choice)
            return torch.multinomial(probs, 1)

        for _ in range(max_new_tokens):
            nxt = _pick(logits)                                # [n, 1]
            if eos_id is not None:
                finished = finished | (nxt.squeeze(-1) == eos_id)
                nxt = torch.where(finished.unsqueeze(-1),
                                  torch.full_like(nxt, pad_id), nxt)
            gen_cols.append(nxt)
            if bool(finished.all()):
                break
            step_pos = next_pos.unsqueeze(-1)                  # [n, 1]
            next_pos = next_pos + 1
            full_mask = torch.cat(
                [full_mask, (~finished).long().unsqueeze(-1).to(full_mask.dtype)], dim=-1)
            out = hf(input_ids=nxt, attention_mask=full_mask, position_ids=step_pos,
                     past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]

        if not gen_cols:
            return ["" for _ in range(n)]
        gen = torch.cat(gen_cols, dim=1)                       # [n, T]
        texts: List[str] = []
        for row in gen:
            toks = row.tolist()
            if eos_id is not None and eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            toks = [t for t in toks if t != pad_id]
            texts.append(self.tokenizer.decode(toks, skip_special_tokens=True).strip())
        return texts

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        # Fall back to the single HF model / device when there is no separate
        # second HF model (i.e. the plain non-vLLM path, e.g. CPU/MPS).
        hf = getattr(self, "HF_model", None) or self.model
        hf_dev = getattr(self, "HF_device", None) or self.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=hf_dev)
        else:
            attention_mask = attention_mask.to(hf_dev)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = hf(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = hf(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

