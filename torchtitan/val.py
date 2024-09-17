import time
from pathlib import Path
import gc
import torch
from torch.nn.functional import cross_entropy

def loss_fn(pred, labels):
    return cross_entropy(
        pred.flatten(0, 1), labels.flatten(0, 1)
    )

def validate(
    model,
    data_loader,
    eval_state,
    logger,
    metric_logger,
    parallel_dims, 
    gc_handler,
    train_context,
    gpu_memory_monitor,
    data_loading_times,
    time_last_log,
    freq, 
    color,
    train_step,
    num_flop_per_token,
    gpu_peak_flops,
    dp_rank,
    fin_val_store,
):
    n_tokens = 0
    total_loss = 0
    total_perplexity = 0
    cnt = 0
    model.eval()
    eval_state.step = 0
    val_data_iterator = iter(data_loader)
    while True:
        if fin_val_store.num_keys() > 1:
            batch = None
        batch = next(val_data_iterator,None)
        if not batch:
            fin_val_store.set(str(dp_rank),"valfin")
            logger.info("plan to exit")
            fin_val_store.wait(["0","1"])
            time.sleep(0.2)
            logger.info("exiting")
            break
        eval_state.step += 1
        data_load_start = time.perf_counter()

        input_ids, labels = batch
        n_tokens_in_curr = labels.numel()
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        with train_context():
            with torch.no_grad():
                if fin_val_store.num_keys() > 1:
                    continue
                else:
                    n_tokens = n_tokens_in_curr
                    data_loading_times.append(time.perf_counter() - data_load_start)
                    pred = model(input_ids)
                    loss = loss_fn(pred, labels)
                    del pred
        time_delta = time.perf_counter() - time_last_log
        total_loss+=loss
        total_perplexity+=2**loss
        cnt+=1
        wps = n_tokens / (
            time_delta * parallel_dims.model_parallel_size
        )
        mfu = 100 * num_flop_per_token * wps / gpu_peak_flops
        time_end_to_end = time_delta / freq
        time_data_loading = sum(data_loading_times) / len(data_loading_times)
        time_data_loading_pct = 100 * sum(data_loading_times) / time_delta
        gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
        # metrics = {
        #     "val/loss_metrics/global_avg_loss": loss,
        #     "val/loss_metrics/global_avg_perplexity": perplexity,
        #     "val/wps": wps,
        #     "val/mfu(%)": mfu,
        #     "val/time_metrics/end_to_end(s)": time_end_to_end,
        #     "val/time_metrics/data_loading(s)": time_data_loading,
        #     "val/time_metrics/data_loading(%)": time_data_loading_pct,
        #     "val/memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
        #     "val/memory/max_active(%)": gpu_mem_stats.max_active_pct,
        #     "val/memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
        #     "val/memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
        #     "val/memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
        #     "val/memory/num_ooms": gpu_mem_stats.num_ooms,
        # }
        logger.info(
            "context: val"
            f"{color.cyan}step: {eval_state.step}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}wps: {round(wps):,}  "
            f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
        )
        time_last_log = time.perf_counter()
    #metric_logger.log(metrics, step=train_step)


    del val_data_iterator
    del data_loader
    if loss:
        del loss
    

    model.train()
