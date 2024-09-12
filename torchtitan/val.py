import time
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
    gpu_peak_flops
):
    # n_tokens = 0
    # total_loss = 0
    # total_perplexity = 0
    # cnt = 0
    eval_state.step = 0
    logger.info("in valid")
    for batch in data_loader:
        eval_state.step += 1
        logger.info(eval_state.step)
        # data_load_start = time.perf_counter()

        input_ids, labels = batch
        # n_tokens += labels.numel()
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        # data_loading_times.append(time.perf_counter() - data_load_start)
        with train_context():
            pred = model(input_ids)
            loss = loss_fn(pred, labels)
    del eval_state
    del input_ids
    del loss
    del labels



        # total_loss += loss
        # total_perplexity += 2 ** loss
        # cnt += 1


    # loss = total_loss / cnt
    # perplexity = total_perplexity / cnt
    # eval_state.log_steps.append(train_step)

    # time_delta = time.perf_counter() - time_last_log

    # wps = n_tokens / (
    #     time_delta * parallel_dims.model_parallel_size
    # )

    # mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

    # time_end_to_end = time_delta / freq
    # time_data_loading = sum(data_loading_times) / len(data_loading_times)
    # time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

    # gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

    # metrics = {
    #     "val/loss_metrics/global_avg_loss": loss,
    #     "val/loss_metrics/global_max_loss": loss,
    #     "val/loss_metrics/global_avg_perplexity": perplexity,
    #     "val/loss_metrics/global_max_perplexity": perplexity,
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
    # metric_logger.log(metrics, step=train_step)
    # print(f"logged in aim at step {train_step}")
    # logger.info(
    #     "context: val  "
    #     f"{color.cyan}step: {train_step}  "
    #     f"{color.green}loss: {loss:7.4f}  "
    #     f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
    #     f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
    #     f"{color.blue}wps: {round(wps):,}  "
    #     f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
    # )

    # data_loading_times.clear()
    # time_last_log = time.perf_counter()
    # gpu_memory_monitor.reset_peak_stats()

    gc.collect(1)
    torch.cuda.empty_cache()    
        # barrier 
