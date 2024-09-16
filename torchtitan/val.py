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
    logger.info("in valid")
    # n_tokens = 0
    # total_loss = 0
    # total_perplexity = 0
    # cnt = 0
    model.eval()
    eval_state.step = 0
    val_data_iterator = iter(data_loader)
    while True:
        batch = next(val_data_iterator,None)
        if fin_val_store.num_keys() > 1:
            batch = None
        if not batch:
            fin_val_store.set(str(dp_rank),"valfin")
            logger.info("plan to exit")
            fin_val_store.wait(["0","1"])
            time.sleep(0.2)
            logger.info("exiting")
            break
        eval_state.step += 1
        logger.info(eval_state.step)
        # data_load_start = time.perf_counter()
        # n_tokens += labels.numel()

        input_ids, labels = batch
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        # data_loading_times.append(time.perf_counter() - data_load_start)
        with train_context():
            with torch.no_grad():
                if fin_val_store.num_keys() > 1:
                    continue
                else:
                    pred = model(input_ids)
                    loss = loss_fn(pred, labels)
                    del pred
    del val_data_iterator
    del data_loader
    del loss
    
    logger.info("all deleted")

    model.train()

    # torch.cuda.empty_cache()    
        # barrier 
